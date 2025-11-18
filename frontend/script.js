document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("uploadForm");
    const fileInput = document.getElementById("fileInput");
    const preview = document.getElementById("preview");
    const heatmapImg = document.getElementById("heatmap");
    const loader = document.getElementById("loader");

    const resultCard = document.getElementById("resultCard");
    const predLabel = document.getElementById("predLabel");
    const predConfidence = document.getElementById("predConfidence");
    const predSeverity = document.getElementById("predSeverity");
    const treatmentTableBody = document.querySelector("#treatmentTable tbody");

    const chatBox = document.getElementById("chatBox");
    const chatInput = document.getElementById("chatInput");
    const chatBtn = document.getElementById("chatBtn");
    
    // Toggle Chatbox Elements
    const chatToggleBtn = document.getElementById("chatToggleBtn");
    const chatContainer = document.getElementById("chatContainer");

    // --- Form submit (Prediction) ---
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const file = fileInput.files[0];
        if (!file) return alert("Select an image!");

        loader.style.display = "block";
        resultCard.style.display = "none";
        heatmapImg.style.display = "none";
        treatmentTableBody.innerHTML = "";

        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch("http://127.0.0.1:5000/predict", { method: "POST", body: formData });
            const data = await res.json();

            // Preview
            preview.src = URL.createObjectURL(file);
            preview.style.display = "block";

            // Grad-CAM
            if (data.gradcam) {
                heatmapImg.src = "data:image/jpeg;base64," + data.gradcam;
                heatmapImg.style.display = "block";
            }

            // Result info
            predLabel.innerText = data.label;
            predConfidence.innerText = (data.confidence * 100).toFixed(2);
            predSeverity.innerText = data.severity;

            // Treatment table
            treatmentTableBody.innerHTML = `<tr><td>${data.label}</td><td>${data.treatment}</td></tr>`;

            resultCard.style.display = "block";

        } catch (err) {
            alert("Error communicating with backend!");
            console.error(err);
        } finally {
            loader.style.display = "none";
        }
    });

    // --- File preview ---
    fileInput.addEventListener("change", () => {
        const file = fileInput.files[0];
        if (file) {
            preview.src = URL.createObjectURL(file);
            preview.style.display = "block";
        }
    });

    // ------------------------------------------------------
    // --- NEW CHATBOT LOGIC (Connects to Python Backend) ---
    // ------------------------------------------------------
    
    // 1. Toggle Chat Window
    if (chatToggleBtn && chatContainer) {
        chatToggleBtn.addEventListener("click", () => {
            // Toggle between 'none' and 'flex'
            if (chatContainer.style.display === "none" || chatContainer.style.display === "") {
                chatContainer.style.display = "flex";
            } else {
                chatContainer.style.display = "none";
            }
        });
    }

    // 2. Send Message
    chatBtn.addEventListener("click", async () => {
        const msg = chatInput.value.trim();
        if (!msg) return;

        // Show User Message
        appendChat("You", msg);
        chatInput.value = "";

        // Show "Thinking..."
        const loadingId = "loading-" + Date.now();
        appendChat("Bot", "<i>Thinking...</i>", loadingId);

        try {
            // Send to Python Backend
            const res = await fetch("http://127.0.0.1:5000/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: msg })
            });

            const data = await res.json();

            // Remove "Thinking..." and show response
            removeChat(loadingId);
            
            if (data.reply) {
                appendChat("Bot", data.reply);
            } else if (data.error) {
                appendChat("Bot", "Error: " + data.error);
            }

        } catch (err) {
            console.error(err);
            removeChat(loadingId);
            appendChat("Bot", "Error: Could not connect to server.");
        }
    });

    function appendChat(sender, message, id = null) {
        const p = document.createElement("p");
        if (id) p.id = id;
        
        // Different colors for User vs Bot
        if (sender === "You") {
            p.style.textAlign = "right";
            p.style.backgroundColor = "#4caf50"; 
            p.style.color = "white";
            p.style.padding = "8px";
            p.style.borderRadius = "10px";
            p.style.margin = "5px 0 5px auto"; // Align right
            p.style.maxWidth = "80%";
        } else {
            p.style.textAlign = "left";
            p.style.backgroundColor = "#444";
            p.style.color = "white";
            p.style.padding = "8px";
            p.style.borderRadius = "10px";
            p.style.margin = "5px auto 5px 0"; // Align left
            p.style.maxWidth = "80%";
        }

        p.innerHTML = `<strong>${sender}:</strong> ${message}`;
        chatBox.appendChild(p);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function removeChat(id) {
        const element = document.getElementById(id);
        if (element) {
            element.remove();
        }
    }

    // --- Retrain button ---
    const retrainBtn = document.getElementById("retrainBtn");
    const retrainStatus = document.getElementById("retrainStatus");

    if(retrainBtn){
        retrainBtn.addEventListener("click", async () => {
            retrainStatus.innerText = "⏳ Retraining started...";
            try {
                const res = await fetch("http://127.0.0.1:5000/retrain", { method: "POST" });
                const data = await res.json();
                if (res.ok) {
                    retrainStatus.innerText = "✅ " + data.message;
                } else {
                    retrainStatus.innerText = "❌ Retraining failed: " + (data.error || "unknown error");
                }
            } catch (err) {
                retrainStatus.innerText = "❌ Error connecting to backend.";
                console.error(err);
            }
        });
    }
});