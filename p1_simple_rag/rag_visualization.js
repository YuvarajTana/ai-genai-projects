// Tab switching functionality
function showTab(tabName) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    
    // Remove active class from all tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Show selected tab content
    document.getElementById(tabName).classList.add('active');
    
    // Add active class to clicked tab
    event.target.classList.add('active');
}

// Animation functions
let animationInterval = null;

function animateIndexing() {
    resetAnimation();
    const output = document.getElementById('demo-output');
    const boxes = document.querySelectorAll('#indexing-flow .flow-box');
    
    output.textContent = 'Starting indexing process...\n\n';
    
    let step = 0;
    const steps = [
        {
            step: 1,
            message: '📄 Step 1: Loading documents from ./data directory...\n   - Scanning for .txt, .pdf, .docx, .md files\n   - Found documents: document1.txt, guide.pdf\n',
            delay: 2000
        },
        {
            step: 2,
            message: '✂️ Step 2: Chunking documents...\n   - document1.txt: 2000 chars → 3 chunks (900 chars each, 120 overlap)\n   - guide.pdf: 5000 chars → 6 chunks\n   - Total: 9 chunks created\n',
            delay: 2000
        },
        {
            step: 3,
            message: '🧮 Step 3: Generating embeddings...\n   - Using SentenceTransformer (all-MiniLM-L6-v2)\n   - Converting 9 chunks to 384-dimensional vectors\n   - Normalizing embeddings...\n',
            delay: 2500
        },
        {
            step: 4,
            message: '💾 Step 4: Storing in ChromaDB...\n   - Collection: "docs"\n   - Storing chunks with metadata (source, chunk index)\n   - ✅ Indexing complete! 9 chunks indexed.\n',
            delay: 2000
        }
    ];
    
    function executeStep(index) {
        if (index >= steps.length) {
            output.textContent += '\n✅ Indexing process completed successfully!';
            return;
        }
        
        const currentStep = steps[index];
        const box = Array.from(boxes).find(b => parseInt(b.dataset.step) === currentStep.step);
        
        // Highlight current box
        boxes.forEach(b => b.classList.remove('active'));
        if (box) {
            box.classList.add('active');
        }
        
        // Update output
        output.textContent += currentStep.message;
        output.scrollTop = output.scrollHeight;
        
        // Move to next step
        setTimeout(() => {
            executeStep(index + 1);
        }, currentStep.delay);
    }
    
    executeStep(0);
}

function animateQuerying() {
    resetAnimation();
    const output = document.getElementById('demo-output');
    const boxes = document.querySelectorAll('#querying-flow .flow-box');
    
    output.textContent = 'Starting query process...\n\n';
    
    let step = 0;
    const steps = [
        {
            step: 1,
            message: '❓ Step 1: User Query\n   Query: "What is machine learning?"\n',
            delay: 1500
        },
        {
            step: 2,
            message: '🧮 Step 2: Query Embedding\n   - Converting query to vector using SentenceTransformer\n   - Query vector: [0.234, -0.567, ..., 0.123] (384 dimensions)\n',
            delay: 2000
        },
        {
            step: 3,
            message: '🔍 Step 3: Similarity Search\n   - Searching ChromaDB for top-4 similar chunks\n   - Computing cosine similarity...\n   - Results:\n     • document.txt::chunk_5 (distance: 0.12)\n     • guide.pdf::chunk_2 (distance: 0.23)\n     • notes.md::chunk_1 (distance: 0.31)\n     • doc.txt::chunk_8 (distance: 0.45)\n',
            delay: 2500
        },
        {
            step: 4,
            message: '📝 Step 4: Building Prompt\n   - Combining retrieved contexts\n   - Formatting prompt with instructions\n   - Context length: ~3600 characters\n',
            delay: 2000
        },
        {
            step: 5,
            message: '🤖 Step 5: LLM Generation\n   - Sending prompt to Ollama API (gpt-oss:20b)\n   - Generating response based on context...\n',
            delay: 2500
        },
        {
            step: 6,
            message: '💬 Step 6: Answer Generated\n   Answer: "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed..."\n\n   Sources:\n   - document.txt (chunk 5, distance: 0.12)\n   - guide.pdf (chunk 2, distance: 0.23)\n',
            delay: 2000
        }
    ];
    
    function executeStep(index) {
        if (index >= steps.length) {
            output.textContent += '\n✅ Query process completed!';
            return;
        }
        
        const currentStep = steps[index];
        const box = Array.from(boxes).find(b => parseInt(b.dataset.step) === currentStep.step);
        
        // Highlight current box
        boxes.forEach(b => b.classList.remove('active'));
        if (box) {
            box.classList.add('active');
        }
        
        // Update output
        output.textContent += currentStep.message + '\n';
        output.scrollTop = output.scrollHeight;
        
        // Move to next step
        setTimeout(() => {
            executeStep(index + 1);
        }, currentStep.delay);
    }
    
    executeStep(0);
}

function resetAnimation() {
    // Clear any running animations
    if (animationInterval) {
        clearInterval(animationInterval);
        animationInterval = null;
    }
    
    // Remove active class from all boxes
    document.querySelectorAll('.flow-box').forEach(box => {
        box.classList.remove('active');
    });
    
    // Clear output
    document.getElementById('demo-output').textContent = 'Click a button above to see the flow animation...';
}

// Add click handlers to flow boxes for interactive exploration
document.addEventListener('DOMContentLoaded', function() {
    const boxes = document.querySelectorAll('.flow-box');
    
    boxes.forEach(box => {
        box.addEventListener('click', function() {
            // Remove active from siblings in same flow
            const parent = this.closest('.flow-diagram');
            if (parent) {
                parent.querySelectorAll('.flow-box').forEach(b => {
                    b.classList.remove('active');
                });
            }
            // Toggle active on clicked box
            this.classList.toggle('active');
        });
    });
    
    // Add smooth scroll behavior
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
});

// Add keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Press 'i' to animate indexing
    if (e.key === 'i' || e.key === 'I') {
        animateIndexing();
    }
    // Press 'q' to animate querying
    if (e.key === 'q' || e.key === 'Q') {
        animateQuerying();
    }
    // Press 'r' to reset
    if (e.key === 'r' || e.key === 'R') {
        resetAnimation();
    }
});