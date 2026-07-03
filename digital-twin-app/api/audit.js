import { Pinecone } from '@pinecone-database/pinecone';
import { GoogleGenerativeAI } from '@google/generative-ai';

// A simple function to pause the code
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

export default async function handler(req, res) {
    // Only allow POST requests
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method not allowed' });
    }

    const userPrompt = req.body.prompt;
    
    // Connect to Vercel's secure environment variables
    const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    
    // Target the specific Pinecone bookshelf you created
    const index = pinecone.index("digital-twin");

    try {
        // =================================================================
        // THE SETUP MODE (Run this once to fill the bookshelf)
        // =================================================================
        if (userPrompt === "INIT_DB") {
            const lawsDocUrl = "https://docs.google.com/document/d/1sQYqfa4gJ8bm2k7DsOuCF9ics8MjY9AXsYMjLv6fzLw/export?format=txt";
            const rawDataDocUrl = "https://drive.google.com/uc?export=download&id=176KSKzJ6a1fx5c_eofeZTIE3D5oP3aLI";

            const [lawsRes, rawRes] = await Promise.all([fetch(lawsDocUrl), fetch(rawDataDocUrl)]);
            const allText = (await lawsRes.text()) + "\n\n" + (await rawRes.text());

            // Chop the massive documents into bite-sized chunks
            const chunks = allText.split('\n\n').filter(c => c.trim().length > 100);
            
            // Using the newest, 3072-dimension flagship model
            const embeddingModel = genAI.getGenerativeModel({ model: "gemini-embedding-001" });

            let vectorsToUpload = [];
            
            // Convert text to math vectors (with Smart Retries!)
            for (let i = 0; i < chunks.length; i++) {
                const chunkText = chunks[i];
                let success = false;
                let retries = 0;

                while (!success && retries < 3) {
                    try {
                        const embedResult = await embeddingModel.embedContent(chunkText);
                        
                        vectorsToUpload.push({
                            id: `chunk_${i}`,
                            values: embedResult.embedding.values,
                            metadata: { text: chunkText }
                        });
                        
                        success = true; // It worked, break the while loop
                        
                        // A smaller, normal speed bump to be polite
                        await delay(2000); 

                    } catch (error) {
                        // If Google throws the 429 error, catch it and take a deep breath
                        if (error.status === 429) {
                            console.log(`Chunk ${i} hit a speed limit! Pausing for 15 seconds...`);
                            await delay(15000); 
                            retries++; // Count the strike
                        } else {
                            // If it's a completely different error, crash normally
                            throw error; 
                        }
                    }
                }
            }

            // Upload everything to Pinecone
            await index.upsert(vectorsToUpload);
            return res.status(200).json({ reply: `Success! Uploaded ${vectorsToUpload.length} blocks of knowledge to Pinecone.` });
        }

        // =================================================================
        // THE RETRIEVAL MODE (The everyday smart librarian)
        // =================================================================
        
        // 1. Turn your new question into a math vector
        const embeddingModel = genAI.getGenerativeModel({ model: "gemini-embedding-001" });
        const promptEmbed = await embeddingModel.embedContent(userPrompt);

        // 2. Search Pinecone for the 3 most mathematically relevant pieces of data
        const searchResults = await index.query({
            vector: promptEmbed.embedding.values,
            topK: 3,
            includeMetadata: true
        });

        // 3. Extract the actual text from those top 3 results
        const relevantContext = searchResults.matches.map(match => match.metadata.text).join("\n\n---\n\n");

        // 4. Build the micro-payload for Gemini
        const systemInstruction = `You are a strategic business advisor. 
You are given specific network scaling laws as context, but you must make them practical.

RULE: Never use jargon without explaining it in one simple sentence.
STRUCTURE:
1. THE INSIGHT: Summarize the advice in one plain-English sentence.
2. THE LOGIC (The "Why"): Briefly explain the law (e.g., "Zipf's Law says small things matter as much as big ones").
3. THE ACTION: Give the user 2 concrete, step-by-step things to do today.

CONTEXT: 
${relevantContext}`;

        // 5. Send the tiny payload to Gemini 2.5 Flash
        const aiModel = genAI.getGenerativeModel({ 
            model: "gemini-2.5-flash",
            systemInstruction: systemInstruction 
        });

        const finalResult = await aiModel.generateContent(userPrompt);
        
        return res.status(200).json({ reply: finalResult.response.text() });

    } catch (error) {
        console.error("CRITICAL FAILURE:", error);
        // Force the app to send the EXACT error message to the browser, not just "System Error"
        return res.status(500).json({ 
            error: error.message, 
            stack: error.stack 
        });
    }
} // <--- Added this vital final brace to close out the handler function!
