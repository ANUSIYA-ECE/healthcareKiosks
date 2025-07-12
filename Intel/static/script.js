document.getElementById("symptomForm").addEventListener("submit", function (e) {
    e.preventDefault();
    const symptoms = document.getElementById("symptoms").value;

    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/x-www-form-urlencoded",
        },
        body: "symptoms=" + encodeURIComponent(symptoms),
    })
    .then((response) => response.json())
    .then((data) => {
        document.getElementById("responseBox").textContent = data.reply;
    })
    .catch((error) => {
        document.getElementById("responseBox").textContent = "Error: " + error;
    });
});
