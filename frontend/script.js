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

    // --- Rule-based chatbot responses ---
    const chatbotRules = {
        "leaf curl": "Leaf Curl can be treated by removing infected leaves, using neem oil, and ensuring airflow.",
        "leaf spot": "Leaf Spot requires pruning infected leaves and applying fungicide.",
        "powdery mildew": "Powdery Mildew can be treated with wettable sulphur or potassium bicarbonate sprays.",
        "whitefly": "Whitefly infestation can be controlled with sticky traps, neem oil, or insecticidal soap.",
        "yellowish": "Yellowish leaves may indicate nutrient deficiency; apply balanced fertilizer.",
        "healthy": "The plant is healthy. Maintain regular care, water properly, and monitor for pests.",
        "watering": "Ensure proper watering: not too much, not too little. Check soil moisture regularly.",
        "fertilizer": "Use a balanced fertilizer or compost to boost plant health and prevent deficiencies.",
        "disease prevention": "Keep the field clean, remove infected leaves, rotate crops, and monitor plants weekly.",
        "pest control": "Use neem oil, sticky traps, or organic insecticides to manage pests naturally.",
        "general advice": "Provide sufficient sunlight, water correctly, and monitor for symptoms regularly."
    };

    // --- Form submit ---
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
            console.log(data);

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

    // --- Chatbot ---
    chatBtn.addEventListener("click", () => {
        const msg = chatInput.value.trim().toLowerCase();
        if (!msg) return;
        appendChat("You", chatInput.value);
        chatInput.value = "";

        let response = "Sorry, I don't understand. Try asking about a disease or care tip.";
        for (const key in chatbotRules) {
            if (msg.includes(key)) {
                response = chatbotRules[key];
                break;
            }
        }
        appendChat("Bot", response);
    });

    function appendChat(sender, message) {
        const p = document.createElement("p");
        p.innerHTML = `<strong>${sender}:</strong> ${message}`;
        chatBox.appendChild(p);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // --- Retrain button ---
    const retrainBtn = document.getElementById("retrainBtn");
    const retrainStatus = document.getElementById("retrainStatus");

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
});
