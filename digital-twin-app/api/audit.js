import { Pinecone } from '@pinecone-database/pinecone';
import { GoogleGenerativeAI } from '@google/generative-ai';

// A simple function to pause the code and prevent 429 Rate Limit errors
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
            
            // Convert text to math vectors (with a speed limit!)
            for (let i = 0; i < chunks.length; i++) {
                const chunkText = chunks[i];
                const embedResult = await embeddingModel.embedContent(chunkText);
                
                vectorsToUpload.push({
                    id: `chunk_${i}`,
                    values: embedResult.embedding.values,
                    metadata: { text: chunkText }
                });
                
                // The Magic Fix: Pause for 1.5 seconds between each request to Google
                await delay(4500); 
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
        const systemInstruction = `You are a highly strategic Digital Twin. 
        Use ONLY the following retrieved context to answer the user's prompt. Do not hallucinate outside information.
        
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
        return res.status(500).json({ error: "System Error. Check Vercel Logs." });
    }
}
