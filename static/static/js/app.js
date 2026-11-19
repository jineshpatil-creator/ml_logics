document.addEventListener("DOMContentLoaded", () => {

    const form = document.getElementById("form");
    const button = document.getElementById("submit");
    const banner = document.getElementById("banner");
    const result = document.getElementById("result");

    if (!form) return;

    const codeFields = [
        "Major_Category",
        "Year_of_Study",
        "Primary_Use_Case",
        "Paid_Subscription",
        "Institutional_Policy"
    ];

    form.addEventListener("submit", async (event) => {

        // Prevent normal GET form submission
        event.preventDefault();

        banner.hidden = true;
        result.hidden = true;

        document.querySelectorAll(".field").forEach(field => {
            field.classList.remove("invalid");
            field.querySelector(".err").textContent = "";
        });

        const payload = {};
        let valid = true;

        const inputs = form.querySelectorAll("input");

        inputs.forEach(input => {

            const value = input.value.trim();
            const field = input.closest(".field");
            const error = field.querySelector(".err");

            if (!value) {
                error.textContent = "This field is required.";
                field.classList.add("invalid");
                valid = false;
                return;
            }

            const number = Number(value);

            if (!Number.isFinite(number)) {
                error.textContent = "Enter a valid number.";
                field.classList.add("invalid");
                valid = false;
                return;
            }

            if (codeFields.includes(input.name) &&
                !Number.isInteger(number)) {

                error.textContent = "Enter a whole-number category code.";
                field.classList.add("invalid");
                valid = false;
                return;
            }

            payload[input.name] = number;
        });

        if (!valid) return;

        button.disabled = true;
        button.classList.add("loading");
        button.querySelector("span").textContent = "Predicting...";

        try {

            const response = await fetch("/api/predict", {

                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(payload)

            });

            const data = await response.json();

            if (!response.ok) {

                if (data.fields) {
                    Object.entries(data.fields).forEach(([name, message]) => {
                        const field = form.querySelector(
                            `[name="${name}"]`
                        )?.closest(".field");

                        if (field) {
                            field.classList.add("invalid");
                            field.querySelector(".err").textContent = message;
                        }
                    });
                }

                throw new Error(data.error || "Prediction failed.");
            }

            // Display predicted class
            document.getElementById("class-code").textContent =
                data.predicted_class;

            document.getElementById("class-echo").textContent =
                data.predicted_class;

            // Display probabilities
            const bars = document.getElementById("bars");
            bars.innerHTML = "";

            if (data.probabilities) {

                const highest = Math.max(
                    ...data.probabilities.map(item => item.probability)
                );

                data.probabilities.forEach(item => {

                    const percentage = item.probability * 100;

                    const li = document.createElement("li");

                    if (item.probability === highest) {
                        li.classList.add("top");
                    }

                    const label = document.createElement("span");
                    label.textContent = `Class ${item.class}`;

                    const track = document.createElement("span");
                    track.className = "track";

                    const fill = document.createElement("span");
                    fill.className = "fill";
                    fill.style.width = `${percentage}%`;

                    track.appendChild(fill);

                    const value = document.createElement("span");
                    value.textContent = `${percentage.toFixed(1)}%`;

                    li.append(label, track, value);
                    bars.appendChild(li);
                });
            }

            result.hidden = false;
            result.classList.remove("show");

            requestAnimationFrame(() => {
                result.classList.add("show");
            });

            result.scrollIntoView({
                behavior: "smooth",
                block: "center"
            });

        } catch (error) {

            banner.textContent = error.message;
            banner.hidden = false;

        } finally {

            button.disabled = false;
            button.classList.remove("loading");
            button.querySelector("span").textContent = "Predict class";

        }

    });

    form.addEventListener("reset", () => {

        banner.hidden = true;
        result.hidden = true;

        document.querySelectorAll(".field").forEach(field => {
            field.classList.remove("invalid");
            field.querySelector(".err").textContent = "";
        });

    });

});
