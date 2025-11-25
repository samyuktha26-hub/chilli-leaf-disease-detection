document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("uploadForm");
    const fileInput = document.getElementById("fileInput");
    const preview = document.getElementById("preview");
    const invalidMsg = document.getElementById("invalidMsg");
    const heatmapImg = document.getElementById("heatmap");
    const loader = document.getElementById("loader");

    const resultCard = document.getElementById("resultCard");
    const predLabel = document.getElementById("predLabel");
    const predConfidence = document.getElementById("predConfidence");
    const predSeverity = document.getElementById("predSeverity");
    const treatmentTableBody = document.querySelector("#treatmentTable tbody");

    // chatbot
    const chatToggleBtn = document.getElementById("chatToggleBtn");
    const chatContainer = document.getElementById("chatContainer");
    const chatBox = document.getElementById("chatBox");
    const chatBtn = document.getElementById("chatBtn");
    const chatInput = document.getElementById("chatInput");

    const alertSound = document.getElementById("alertSound");

    chatToggleBtn.addEventListener("click", () => {
        chatContainer.style.display = chatContainer.style.display === "flex" ? "none" : "flex";
    });

      fileInput.addEventListener("change", () => {
        const file = fileInput.files[0];
        if (file) {
            preview.src = URL.createObjectURL(file);
            preview.style.display = "block";
            invalidMsg.style.display = "none";
            resultCard.style.display = "none";
            heatmapImg.style.display = "none";
        }
    });

    // --- Upload form ---
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const file = fileInput.files[0];
        if (!file) return alert("Select an image");

        preview.src = URL.createObjectURL(file);
        preview.style.display = "block";
        loader.style.display = "block";
        invalidMsg.style.display = "none";
        resultCard.style.display = "none";
        heatmapImg.style.display = "none";
        treatmentTableBody.innerHTML = "";

        const formData = new FormData();
        formData.append("file", file);

        // FIRST — verify if chilli leaf
        const leafCheck = await fetch("http://127.0.0.1:5000/validate_leaf", {
            method: "POST",
            body: formData
        });

        const leafResult = await leafCheck.json();

        if (!leafResult.is_chilli_leaf) {
            loader.style.display = "none";
            invalidMsg.textContent = "❌ Not a chilli leaf. Upload another image.";
            invalidMsg.style.display = "block";

            alertSound.play();

            chatContainer.style.display = "flex";
            appendChat("Bot", "This seems to be not a chilli leaf. Upload a valid chilli leaf image for prediction.");
            return;
        }

        // If valid chilli leaf → send for prediction
        try {
            const res = await fetch("http://127.0.0.1:5000/predict", { method: "POST", body: formData });
            const data = await res.json();

            predLabel.textContent = data.label;
            predConfidence.textContent = (data.confidence * 100).toFixed(2);
            predSeverity.textContent = data.severity;

            treatmentTableBody.innerHTML = `<tr><td>${data.label}</td><td>${data.treatment}</td></tr>`;

            if (data.gradcam) {
                heatmapImg.src = "data:image/jpeg;base64," + data.gradcam;
                heatmapImg.style.display = "block";
            }

            resultCard.style.display = "block";
        } catch (error) {
            alert("Error connecting to backend");
            console.error(error);
        } finally {
            loader.style.display = "none";
        }
    });

    // chatbot send
    chatBtn.addEventListener("click", async () => {
        const msg = chatInput.value.trim();
        if (!msg) return;

        appendChat("You", msg);
        chatInput.value = "";

        const loadingId = "load-" + Date.now();
        appendChat("Bot", "<i>Typing...</i>", loadingId);

        const res = await fetch("http://127.0.0.1:5000/chat", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ message: msg })
        });

        const data = await res.json();
        removeChat(loadingId);
        appendChat("Bot", data.reply);
    });

    function appendChat(sender, message, id = null) {
        const p = document.createElement("p");
        if (id) p.id = id;
        p.innerHTML = `<strong>${sender}:</strong> ${message}`;
        p.style.padding = "6px";
        p.style.background = sender === "You" ? "#4caf50" : "#444";
        p.style.margin = "5px";
        p.style.color = "white";
        p.style.borderRadius = "8px";
        chatBox.appendChild(p);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function removeChat(id) {
        const x = document.getElementById(id);
        if (x) x.remove();
    }
});
