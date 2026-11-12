from flask import Flask, request, render_template
import os
import traceback
from werkzeug.utils import secure_filename


# IMPORT CURRENT PREDICTION FUNCTIONS  
from predict import (
    load_federated_model,
    load_image,
    predict_image,
    generate_gradcam,
    save_gradcam,
    MODEL_PATH,
    DEVICE,
    LABELS
)

app = Flask(__name__)

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)
UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)
GRADCAM_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "gradcam"
)



# CREATE DIRECTORIES
os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    GRADCAM_FOLDER,
    exist_ok=True
)


# ALLOWED IMAGE EXTENSIONS
ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "bmp",
    "tif",
    "tiff"
}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


# LOAD GLOBAL MODEL ONCE
print()
print("=" * 70)
print("LOADING FEDERATED GLOBAL MODEL")
print("=" * 70)

print()
print("Model path:")
print(MODEL_PATH)

print()
print("Device:")
print(DEVICE)

try:

    MODEL = load_federated_model(
        MODEL_PATH
    )

    print()
    print("Global model loaded successfully.")

except Exception as e:

    MODEL = None

    print()
    print("ERROR: Could not load global model.")
    print(str(e))

    traceback.print_exc()



# HOME PAGE
@app.route(
    "/",
    methods=["GET"]
)
def home():

    return render_template(
        "index.html"
    )



# PREDICTION
@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    print()
    print("=" * 70)
    print("PREDICTION REQUEST RECEIVED")
    print("=" * 70)

    try: 
        # CHECK MODEL
        if MODEL is None:

            raise RuntimeError(
                "Global model is not loaded. "
                "Please check results/global_model.pth."
            )

        file = request.files.get(
            "file"
        )

        if file is None:

            print(
                "ERROR: No file received."
            )

            return render_template(
                "index.html",
                error="No file was uploaded."
            )


        if file.filename == "":

            print(
                "ERROR: Empty filename."
            )

            return render_template(
                "index.html",
                error="Please select an image."
            )


        print(
            "Original filename:",
            file.filename
        )


        if not allowed_file(
            file.filename
        ):

            print(
                "ERROR: Unsupported file type."
            )

            return render_template(
                "index.html",
                error=(
                    "Unsupported image format. "
                    "Please upload PNG, JPG, JPEG, "
                    "BMP, TIF or TIFF."
                )
            )


        # ====================================================
        # SECURE FILENAME
        # ====================================================

        filename = secure_filename(
            file.filename
        )

        if not filename:

            return render_template(
                "index.html",
                error="Invalid filename."
            )


        # ====================================================
        # AVOID OLD FILE CONFLICTS
        # ====================================================

        image_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )


        # ====================================================
        # SAVE IMAGE
        # ====================================================

        file.save(
            image_path
        )

        print()
        print("Image saved:")
        print(image_path)


        # ====================================================
        # LOAD IMAGE
        # ====================================================

        print()
        print("Preparing image for model...")

        input_tensor, original_image = load_image(
            image_path
        )

        print(
            "Input tensor shape:",
            tuple(input_tensor.shape)
        )


        # ====================================================
        # RUN PREDICTION
        # ====================================================

        print()
        print("Running retinal prediction...")


        (
            probabilities,
            confidence,
            top_class,
            top_class_index
        ) = predict_image(
            MODEL,
            input_tensor
        )


        # ====================================================
        # CONVERT PROBABILITIES TO PERCENTAGES
        # ====================================================

        predictions = {}

        for i, label in enumerate(LABELS):

            predictions[label] = round(
                float(
                    probabilities[i]
                    * 100.0
                ),
                2
            )


        confidence_percentage = round(
            float(
                confidence
                * 100.0
            ),
            2
        )


        # ====================================================
        # PRINT PREDICTION RESULTS
        # ====================================================

        print()
        print("=" * 70)
        print("RETINAL CONDITION PROBABILITIES")
        print("-" * 70)

        for label, probability in predictions.items():

            print(
                f"{label:<30}"
                f"{probability:>7.2f}%"
            )

        print("-" * 70)

        print(
            f"Confidence: "
            f"{confidence_percentage:.2f}%"
        )

        print(
            f"Top class: "
            f"{top_class}"
        )

        print("=" * 70)


        # ====================================================
        # GENERATE GRAD-CAM
        # ====================================================

        print()
        print("=" * 70)
        print("GENERATING GRAD-CAM")
        print("=" * 70)

        gradcam_filename = (
            "gradcam_"
            + os.path.splitext(filename)[0]
            + ".png"
        )

        gradcam_file_path = os.path.join(
            GRADCAM_FOLDER,
            gradcam_filename
        )


        # ----------------------------------------------------
        # Generate CAM
        # ----------------------------------------------------

        cam = generate_gradcam(
            MODEL,
            input_tensor,
            top_class_index
        )


        print()
        print(
            "Grad-CAM generated successfully."
        )

        print(
            "Grad-CAM shape:",
            cam.shape
        )


        # ----------------------------------------------------
        # Save CAM
        # ----------------------------------------------------

        save_gradcam(
            original_image,
            cam,
            gradcam_file_path
        )


        print()
        print(
            "Grad-CAM saved:"
        )

        print(
            gradcam_file_path
        )


        # ====================================================
        # BROWSER IMAGE URL
        # ====================================================

        image_url = (
            "/static/uploads/"
            + filename
        )


        # ====================================================
        # BROWSER GRAD-CAM URL
        # ====================================================

        gradcam_url = (
            "/static/gradcam/"
            + gradcam_filename
        )


        # ====================================================
        # RENDER RESULT
        # ====================================================

        print()
        print(
            "Rendering result page..."
        )

        print(
            "=" * 70
        )


        return render_template(

            "index.html",

            # ------------------------------------------------
            # Five probabilities
            # ------------------------------------------------

            predictions=predictions,

            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            confidence=confidence_percentage,

            # ------------------------------------------------
            # Uploaded image
            # ------------------------------------------------

            image_path=image_url,

            # ------------------------------------------------
            # Grad-CAM
            # ------------------------------------------------

            gradcam_path=gradcam_url
        )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        print()
        print("=" * 70)
        print("FLASK PREDICTION ERROR")
        print("=" * 70)

        print(
            "Error:",
            str(e)
        )

        traceback.print_exc()

        print("=" * 70)


        return render_template(

            "index.html",

            error=(
                "Prediction error: "
                + str(e)
            )
        )



# RUN FLASK


if __name__ == "__main__":

    print()
    print("=" * 70)
    print("RETINAL CONDITION ANALYSIS")
    print("=" * 70)

    print()
    print("Upload folder:")
    print(UPLOAD_FOLDER)

    print()
    print("Grad-CAM folder:")
    print(GRADCAM_FOLDER)

    print()
    print("Global model:")
    print(MODEL_PATH)

    print()
    print("Website:")
    print("http://127.0.0.1:5000")

    print()
    print("=" * 70)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )