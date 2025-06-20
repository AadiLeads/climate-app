import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load the model once globally
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'efficientnet_bark_classifier.h5')
model = load_model(MODEL_PATH)

class_names  = ['alder', 'beech', 'birch', 'chestnut', 'ginkgo biloba', 'hornbeam', 'horse chestnut',
                'linden', 'oak', 'oriental plane', 'pine', 'spruce']  # example
tree_info = {
    "alder": {
        "climate": "Cool, temperate climates near rivers and wetlands.",
        "india_regions": "Western Himalayas, Kashmir, Himachal Pradesh, Sikkim.",
        "medicinal_uses": "Bark is used for treating fever and inflammation. Tannins have antibacterial properties."
    },
    "beech": {
        "climate": "Cool temperate zones with high humidity.",
        "india_regions": "Western Himalayas, especially in Uttarakhand and Himachal.",
        "medicinal_uses": "Used for antiseptic properties; bark extracts treat skin disorders and ulcers."
    },
    "birch": {
        "climate": "Cold and moist temperate climates.",
        "india_regions": "Higher altitudes of Jammu & Kashmir and Himachal Pradesh.",
        "medicinal_uses": "Used in treating kidney stones, arthritis, and skin rashes."
    },
    "chestnut": {
        "climate": "Temperate climates with well-drained soils.",
        "india_regions": "Himalayan region, especially Uttarakhand and Himachal.",
        "medicinal_uses": "Bark and leaves used to treat coughs and digestive disorders."
    },
    "ginkgo biloba": {
        "climate": "Temperate zones with moderate sunlight.",
        "india_regions": "Cultivated in botanical gardens in North India; not native.",
        "medicinal_uses": "Improves memory and cognitive function, used in treating Alzheimer’s and dementia."
    },
    "hornbeam": {
        "climate": "Cool climates with moist soil.",
        "india_regions": "Western Himalayas and Sikkim.",
        "medicinal_uses": "Used in Bach flower remedies for fatigue and mental exhaustion."
    },
    "horse chestnut": {
        "climate": "Cool temperate climates with rich soils.",
        "india_regions": "Botanical gardens in Himachal, Uttarakhand, and Kashmir.",
        "medicinal_uses": "Seed extract used for treating varicose veins and inflammation."
    },
    "linden": {
        "climate": "Temperate zones with good rainfall.",
        "india_regions": "Rare, seen in managed landscapes like Shimla and Dehradun.",
        "medicinal_uses": "Flowers used as mild sedatives and for treating colds and anxiety."
    },
    "oak": {
        "climate": "Cool to temperate climates, often mountainous.",
        "india_regions": "Extensive in Uttarakhand, Himachal Pradesh, and Sikkim.",
        "medicinal_uses": "Bark treats diarrhea and wounds due to its astringent properties."
    },
    "oriental plane": {
        "climate": "Temperate climates with adequate moisture.",
        "india_regions": "Jammu & Kashmir, Himachal Pradesh.",
        "medicinal_uses": "Used for skin ailments and gastrointestinal issues."
    },
    "pine": {
        "climate": "Cold, dry to moist temperate climates.",
        "india_regions": "Abundant in Himachal Pradesh, Uttarakhand, and Northeast.",
        "medicinal_uses": "Resin used for treating cough, colds, and wounds. Oil is antimicrobial."
    },
    "spruce": {
        "climate": "Cold climates and high altitudes.",
        "india_regions": "Himalayas in Kashmir and Himachal Pradesh.",
        "medicinal_uses": "Bark and resin used to treat respiratory infections and inflammation."
    }
}


def predict_bark(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    class_index = np.argmax(preds)
    class_name = class_names[class_index]

    info = tree_info.get(class_name, {
        'climate': 'Not available',
        'medicinal_uses': 'Not available',
        'india_regions': 'Not available'
    })

    return {
        'tree': class_name,
        'climate': info['climate'],
        'medicinal_uses': info['medicinal_uses'],
        'india_regions': info['india_regions']
    }
