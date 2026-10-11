import numpy as np
import pandas as pd

# Ordered feature list matching the trained Random Forest model
FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "rainfall"]

# Allowed seasons in UI
ALLOWED_SEASONS = ["Kharif", "Rabi", "Zaid", "Summer", "Winter", "Monsoon"]

# Numerical validation bounds
VALIDATION_BOUNDS = {
    "N": {"min": 0.0, "max": 300.0, "label": "Nitrogen (N)", "unit": "mg/kg"},
    "P": {"min": 0.0, "max": 300.0, "label": "Phosphorus (P)", "unit": "mg/kg"},
    "K": {"min": 0.0, "max": 300.0, "label": "Potassium (K)", "unit": "mg/kg"},
    "temperature": {"min": -10.0, "max": 65.0, "label": "Temperature", "unit": "°C"},
    "humidity": {"min": 0.0, "max": 100.0, "label": "Humidity", "unit": "%"},
    "rainfall": {"min": 0.0, "max": 3000.0, "label": "Rainfall", "unit": "mm"},
}

# Dynamic agronomic profiles for crops in the benchmark dataset
CROP_PROFILES = {
    "rice": {
        "category": "Cereal / Food Grain",
        "duration": "100 - 150 days",
        "water_need": "High (Standing water / regular irrigation)",
        "soil_type": "Clayey, alluvial, or loamy soil with good moisture retention",
        "summary": "Rice is a staple Kharif crop requiring abundant moisture, warm climate, and clay-rich soil capable of retaining water."
    },
    "maize": {
        "category": "Cereal Grain / Coarse Grain",
        "duration": "90 - 120 days",
        "water_need": "Moderate (500 - 800 mm)",
        "soil_type": "Well-drained fertile loam or silt loam",
        "summary": "Maize grows well in warm weather with moderate rainfall. Highly responsive to balanced nitrogen and potassium fertilization."
    },
    "chickpea": {
        "category": "Pulse / Legume",
        "duration": "90 - 120 days",
        "water_need": "Low to Moderate (Drought tolerant)",
        "soil_type": "Deep well-drained loamy or clay loam soil",
        "summary": "Chickpea is a key Rabi pulse crop that fixes atmospheric nitrogen into the soil, thriving in cooler, dry maturity periods."
    },
    "kidneybeans": {
        "category": "Pulse / Legume",
        "duration": "100 - 130 days",
        "water_need": "Moderate (Regular moisture without waterlogging)",
        "soil_type": "Light, deep, well-aerated sandy loam to rich silt",
        "summary": "Also known as Rajma, kidney beans prefer mild temperate conditions and nutrient-rich, well-aerated soils."
    },
    "pigeonpeas": {
        "category": "Pulse / Legume",
        "duration": "150 - 200 days",
        "water_need": "Moderate (Deep root system confers drought resistance)",
        "soil_type": "Well-drained loamy to heavy black cotton soils",
        "summary": "Also known as Arhar or Tur, pigeon peas are widely cultivated in rainfed drylands and enrich soil through biological nitrogen fixation."
    },
    "mothbeans": {
        "category": "Pulse / Legume",
        "duration": "75 - 90 days",
        "water_need": "Low (Highly drought-resistant arid pulse)",
        "soil_type": "Sandy, sandy loam, or gravelly arid soil",
        "summary": "Moth bean is an exceptionally resilient arid-region crop, well-suited for hot, dry zones with low rainfall."
    },
    "mungbean": {
        "category": "Pulse / Legume",
        "duration": "60 - 75 days",
        "water_need": "Low to Moderate",
        "soil_type": "Fertile, well-drained sandy loam or alluvial soil",
        "summary": "Mung bean (Green Gram) is a short-duration pulse often cultivated as a catch or inter-crop in Kharif and Zaid seasons."
    },
    "blackgram": {
        "category": "Pulse / Legume",
        "duration": "75 - 90 days",
        "water_need": "Moderate",
        "soil_type": "Heavier soils with good water-holding capacity, loams",
        "summary": "Black gram (Urad) thrives in warm, humid weather and improves soil health by returning nitrogen biomass to the field."
    },
    "lentil": {
        "category": "Pulse / Legume",
        "duration": "100 - 130 days",
        "water_need": "Low to Moderate (Cold tolerant)",
        "soil_type": "Alluvial, light loams, and well-drained clay soils",
        "summary": "Lentil (Masoor) is predominantly cultivated as a Rabi season pulse, performing best in cooler growing temperatures."
    },
    "pomegranate": {
        "category": "Horticultural Fruit",
        "duration": "Perennial shrub / tree (Matures in 5-7 months per crop cycle)",
        "water_need": "Low to Moderate (Tolerant of semi-arid conditions)",
        "soil_type": "Well-drained light loamy to gravelly soil",
        "summary": "Pomegranate thrives in semi-arid and subtropical regions with hot, dry summers and mild winters."
    },
    "banana": {
        "category": "Horticultural Fruit / Plantation",
        "duration": "10 - 12 months",
        "water_need": "High (Constant moisture, 1500 - 2000 mm)",
        "soil_type": "Rich, deep, fertile loamy soil with high organic matter",
        "summary": "Banana is a moisture-loving, heavy-feeding tropical crop requiring rich organic nutrients, particularly Potassium (K)."
    },
    "mango": {
        "category": "Perennial Fruit Tree",
        "duration": "Perennial (Harvest cycle ~4-5 months from flowering)",
        "water_need": "Moderate (Requires dry dry-spell for flowering)",
        "soil_type": "Deep alluvial, well-drained loamy soil",
        "summary": "Known as the king of fruits, mango demands tropical warmth, clear dry weather during bloom, and well-drained soil."
    },
    "grapes": {
        "category": "Horticultural Fruit / Vine",
        "duration": "Perennial vine (Fruit cycle ~120 - 150 days)",
        "water_need": "Moderate (Requires dry conditions during ripening)",
        "soil_type": "Well-drained sandy loam, gravelly, or red sandy loam",
        "summary": "Grapes perform best in semi-arid climates with warm days and dry periods during cluster development to prevent mildew."
    },
    "watermelon": {
        "category": "Fruit / Cucurbit (Zaid crop)",
        "duration": "80 - 100 days",
        "water_need": "Moderate (Regular early moisture, dry at maturity)",
        "soil_type": "Sandy, sandy loam with good drainage and warmth",
        "summary": "Watermelon is a classic warm-weather Zaid crop that thrives in sandy soils under abundant sunshine and high temperatures."
    },
    "muskmelon": {
        "category": "Fruit / Cucurbit (Zaid crop)",
        "duration": "70 - 90 days",
        "water_need": "Moderate (Controlled irrigation)",
        "soil_type": "Well-drained sandy loam soil rich in organic matter",
        "summary": "Muskmelon demands dry, warm weather during maturity to build natural fruit brix (sweetness) and prevent vine rot."
    },
    "apple": {
        "category": "Temperate Fruit Tree",
        "duration": "Perennial (Fruit maturation ~120 - 180 days)",
        "water_need": "Moderate to High (1000 - 1250 mm distributed)",
        "soil_type": "Deep, well-drained loamy soil with good organic content",
        "summary": "Apple requires a temperate mountain climate with winter chilling hours (freezing dormancy) and cool summer temperatures."
    },
    "orange": {
        "category": "Citrus Fruit",
        "duration": "Perennial tree (Fruit maturation ~7 - 9 months)",
        "water_need": "Moderate (Regular scheduled irrigation)",
        "soil_type": "Light, well-drained loam or alluvial soil",
        "summary": "Oranges grow well in subtropical to tropical climates with distinct dry and wet periods and excellent subsoil drainage."
    },
    "papaya": {
        "category": "Tropical Fruit",
        "duration": "9 - 12 months to first harvest",
        "water_need": "Moderate (Sensitive to waterlogging)",
        "soil_type": "Rich, well-drained sandy loam or alluvial soil",
        "summary": "Papaya is a fast-growing tropical fruit crop requiring steady warm temperatures, sunshine, and well-drained beds."
    },
    "coconut": {
        "category": "Plantation / Cash Crop",
        "duration": "Perennial palm (Continual harvest throughout the year)",
        "water_need": "High (1500 - 2500 mm rainfall or irrigation)",
        "soil_type": "Coastal sandy loam, alluvial, and red loam",
        "summary": "Coconut is a classic coastal palm that demands warm, humid tropical conditions and high potassium availability in soil."
    },
    "cotton": {
        "category": "Fiber / Cash Crop",
        "duration": "150 - 180 days",
        "water_need": "Moderate (500 - 800 mm)",
        "soil_type": "Deep black cotton soils (regur) or fertile alluvial soils",
        "summary": "Cotton requires warm days, abundant sunlight, and frost-free weather, growing exceptionally well in moisture-retentive black soil."
    },
    "jute": {
        "category": "Fiber / Cash Crop",
        "duration": "120 - 150 days",
        "water_need": "High (Pre-monsoon and monsoon showers, 1200+ mm)",
        "soil_type": "Fertile new alluvial soil along river floodplains",
        "summary": "Jute is a major biodegradable natural fiber crop requiring warm, humid weather and standing water for retting processes."
    },
    "coffee": {
        "category": "Beverage / Plantation Crop",
        "duration": "Perennial shrub (Harvest cycle ~7 - 9 months)",
        "water_need": "Moderate to High (1500 - 2000 mm distributed rainfall)",
        "soil_type": "Deep, porous, humus-rich volcanic or forest loams on slopes",
        "summary": "Coffee thrives under shade in cool tropical highland microclimates with well-distributed rain and high relative humidity."
    }
}


def get_crop_profile(crop_name):
    """
    Returns agronomic facts and profile for a given crop name.
    Dynamic and decision-support oriented.
    """
    key = str(crop_name).strip().lower()
    if key in CROP_PROFILES:
        return CROP_PROFILES[key]
    return {
        "category": "Agricultural Crop",
        "duration": "Variable depending on variety",
        "water_need": "Moderate based on local climate",
        "soil_type": "Well-drained fertile agricultural soil",
        "summary": f"{crop_name} is suitable for cultivation under the recommended soil and climate conditions."
    }


def validate_and_preprocess_input(form_data):
    """
    Validates form data and extracts ordered features for the ML model.
    Also handles optional contextual location metadata (which is NOT passed to the ML model).

    Returns:
        tuple: (features_df, cleaned_summary, error_message)
        If valid: (pandas DataFrame shape (1, 6), dict of cleaned inputs, None)
        If invalid: (None, None, error_message)
    """
    errors = []
    cleaned_values = {}

    # 1. Validate numerical features required by the trained model
    for feature in FEATURE_COLUMNS:
        raw_val = form_data.get(feature)

        if raw_val is None or str(raw_val).strip() == "":
            errors.append(f"{VALIDATION_BOUNDS[feature]['label']} is required.")
            continue

        try:
            val = float(str(raw_val).strip())
        except (ValueError, TypeError):
            errors.append(f"{VALIDATION_BOUNDS[feature]['label']} must be a valid number.")
            continue

        bounds = VALIDATION_BOUNDS[feature]
        if val < bounds["min"]:
            errors.append(
                f"{bounds['label']} cannot be less than {bounds['min']} {bounds['unit']}."
            )
        elif val > bounds["max"]:
            errors.append(
                f"{bounds['label']} cannot exceed {bounds['max']} {bounds['unit']}."
            )
        else:
            cleaned_values[feature] = round(val, 2)

    # 2. Validate Season (user-provided metadata, not passed to model)
    season = form_data.get("season", "").strip()
    if not season:
        errors.append("Please select a season.")
    elif season not in ALLOWED_SEASONS:
        errors.append(f"Invalid season selected. Choose one of: {', '.join(ALLOWED_SEASONS)}.")
    else:
        cleaned_values["season"] = season

    # 3. Handle OPTIONAL Location metadata
    # NOTE: Location is strictly contextual for the user and future integrations.
    # It is NEVER passed into the trained Random Forest model.
    location_mode = form_data.get("location_mode", "").strip()
    location_info = None

    if location_mode == "gps":
        lat_raw = form_data.get("latitude", "").strip()
        lon_raw = form_data.get("longitude", "").strip()
        if lat_raw and lon_raw:
            try:
                lat = round(float(lat_raw), 5)
                lon = round(float(lon_raw), 5)
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    location_info = {
                        "mode": "GPS Geolocation",
                        "latitude": lat,
                        "longitude": lon,
                        "display": f"{lat}° N/S, {lon}° E/W"
                    }
            except (ValueError, TypeError):
                pass
    elif location_mode == "manual":
        village = form_data.get("village", "").strip()[:80]
        district = form_data.get("district", "").strip()[:80]
        state = form_data.get("state", "").strip()[:80]
        parts = [p for p in [village, district, state] if p]
        if parts:
            location_info = {
                "mode": "Manual Entry",
                "village": village or "Not specified",
                "district": district or "Not specified",
                "state": state or "Not specified",
                "display": ", ".join(parts)
            }

    cleaned_values["location"] = location_info

    if errors:
        return None, None, errors

    # 4. Build ordered feature vector strictly in the order expected by the model
    # [N, P, K, temperature, humidity, rainfall]
    ordered_features = [
        cleaned_values["N"],
        cleaned_values["P"],
        cleaned_values["K"],
        cleaned_values["temperature"],
        cleaned_values["humidity"],
        cleaned_values["rainfall"],
    ]

    features_df = pd.DataFrame([ordered_features], columns=FEATURE_COLUMNS)

    return features_df, cleaned_values, None
