// SmartCrop AI - Client-side validation, geolocation handling, and interactive UX

document.addEventListener("DOMContentLoaded", function () {
    const cropForm = document.getElementById("cropForm");

    // =========================================================================
    // 1. OPTIONAL LOCATION HANDLERS
    // =========================================================================
    const btnTabGPS = document.getElementById("btnTabGPS");
    const btnTabManual = document.getElementById("btnTabManual");
    const panelGPS = document.getElementById("panelGPS");
    const panelManual = document.getElementById("panelManual");
    const locationModeInput = document.getElementById("location_mode");
    const detectLocationBtn = document.getElementById("detectLocationBtn");
    const clearLocationBtn = document.getElementById("clearLocationBtn");
    const gpsStatus = document.getElementById("gpsStatus");
    const coordsDisplay = document.getElementById("coordsDisplay");
    const latInput = document.getElementById("latitude");
    const lonInput = document.getElementById("longitude");

    // Tab Switching
    if (btnTabGPS && btnTabManual) {
        btnTabGPS.addEventListener("click", function () {
            btnTabGPS.classList.add("active");
            btnTabManual.classList.remove("active");
            panelGPS.style.display = "block";
            panelManual.style.display = "none";
            if (locationModeInput) locationModeInput.value = "gps";
        });

        btnTabManual.addEventListener("click", function () {
            btnTabManual.classList.add("active");
            btnTabGPS.classList.remove("active");
            panelManual.style.display = "block";
            panelGPS.style.display = "none";
            if (locationModeInput) locationModeInput.value = "manual";
        });
    }

    // Geolocation detection - ONLY on explicit button click
    if (detectLocationBtn) {
        detectLocationBtn.addEventListener("click", function () {
            if (!navigator.geolocation) {
                if (gpsStatus) {
                    gpsStatus.textContent = "⚠️ Browser geolocation is not supported on this device. You can enter location manually or continue without location.";
                    gpsStatus.className = "loc-status-msg error";
                }
                return;
            }

            gpsStatus.textContent = "⏳ Requesting GPS location from browser...";
            gpsStatus.className = "loc-status-msg info";
            detectLocationBtn.disabled = true;

            navigator.geolocation.getCurrentPosition(
                function (position) {
                    const lat = position.coords.latitude.toFixed(5);
                    const lon = position.coords.longitude.toFixed(5);

                    if (latInput) latInput.value = lat;
                    if (lonInput) lonInput.value = lon;
                    if (coordsDisplay) coordsDisplay.style.display = "grid";
                    if (clearLocationBtn) clearLocationBtn.style.display = "inline-flex";

                    if (gpsStatus) {
                        gpsStatus.textContent = `✅ Location detected: ${lat}° N/S, ${lon}° E/W. (Recorded for your field reference; not used by the ML model).`;
                        gpsStatus.className = "loc-status-msg success";
                    }
                    detectLocationBtn.disabled = false;
                },
                function (error) {
                    detectLocationBtn.disabled = false;
                    let msg = "⚠️ Geolocation permission was denied or unavailable. You can enter your location manually or proceed without location.";
                    if (error.code === error.TIMEOUT) {
                        msg = "⚠️ Geolocation request timed out. You can enter location manually or proceed without location.";
                    }
                    if (gpsStatus) {
                        gpsStatus.textContent = msg;
                        gpsStatus.className = "loc-status-msg error";
                    }
                },
                { timeout: 10000, enableHighAccuracy: false }
            );
        });
    }

    // Clear Location
    if (clearLocationBtn) {
        clearLocationBtn.addEventListener("click", function () {
            if (latInput) latInput.value = "";
            if (lonInput) lonInput.value = "";
            if (coordsDisplay) coordsDisplay.style.display = "none";
            clearLocationBtn.style.display = "none";
            if (gpsStatus) {
                gpsStatus.textContent = "Location cleared. Prediction will run normally without location.";
                gpsStatus.className = "loc-status-msg";
            }
        });
    }

    // =========================================================================
    // 2. CLIENT-SIDE VALIDATION FOR MODEL INPUTS
    // =========================================================================
    if (!cropForm) return;

    const fields = {
        N: { label: "Nitrogen (N)", min: 0, max: 300, isNumber: true },
        P: { label: "Phosphorus (P)", min: 0, max: 300, isNumber: true },
        K: { label: "Potassium (K)", min: 0, max: 300, isNumber: true },
        temperature: { label: "Temperature", min: -10, max: 65, isNumber: true },
        humidity: { label: "Humidity", min: 0, max: 100, isNumber: true },
        rainfall: { label: "Rainfall", min: 0, max: 3000, isNumber: true },
        season: { label: "Season", isNumber: false }
    };

    function validateField(fieldName) {
        const input = document.getElementById(fieldName);
        const feedback = document.getElementById(`${fieldName}_feedback`);
        if (!input) return true;

        const val = input.value.trim();
        const config = fields[fieldName];
        let error = "";

        if (val === "") {
            error = `${config.label} is required.`;
        } else if (config.isNumber) {
            const num = parseFloat(val);
            if (isNaN(num)) {
                error = `${config.label} must be a valid number.`;
            } else if (num < config.min) {
                error = `${config.label} cannot be less than ${config.min}.`;
            } else if (num > config.max) {
                error = `${config.label} cannot exceed ${config.max}.`;
            }
        }

        if (error) {
            input.style.borderColor = "#e53e3e";
            if (feedback) {
                feedback.textContent = error;
                feedback.style.display = "block";
            }
            return false;
        } else {
            input.style.borderColor = "#38a169";
            if (feedback) {
                feedback.textContent = "";
                feedback.style.display = "none";
            }
            return true;
        }
    }

    // Attach real-time validation on input/change
    Object.keys(fields).forEach(function (fieldName) {
        const elem = document.getElementById(fieldName);
        if (elem) {
            elem.addEventListener("input", function () {
                validateField(fieldName);
            });
            elem.addEventListener("change", function () {
                validateField(fieldName);
            });
        }
    });

    // Form Submission
    cropForm.addEventListener("submit", function (e) {
        let isValid = true;
        Object.keys(fields).forEach(function (fieldName) {
            const fieldValid = validateField(fieldName);
            if (!fieldValid) {
                isValid = false;
            }
        });

        if (!isValid) {
            e.preventDefault();
            // Scroll to first invalid field
            const firstInvalid = document.querySelector(".form-input[style*='rgb(229, 62, 62)'], .form-select[style*='rgb(229, 62, 62)']");
            if (firstInvalid) {
                firstInvalid.focus();
            }
        } else {
            const submitBtn = document.getElementById("predictBtn");
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = "Processing Recommendation...";
            }
        }
    });
});
