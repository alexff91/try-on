import { client } from "@gradio/client";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

// Derive __dirname from import.meta.url
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Read local files
const readLocalFile = (filePath) => {
  return new Promise((resolve, reject) => {
    fs.readFile(filePath, (err, data) => {
      if (err) {
        reject(err);
      } else {
        resolve(new Blob([data]));
      }
    });
  });
};

const run = async () => {
  try {
    // Initialize the Gradio client
    const app = await client("alexff91/FitMirror");

    // Read local image files
    const garmentImage = await readLocalFile(path.join(__dirname, "09236_00.jpg"));
    const humanImage = await readLocalFile(path.join(__dirname, "00121_00.jpg"));

    // Call the /tryon API with the appropriate parameters
    const result = await app.predict("/tryon", [
      {
        "background": humanImage,  // Pass the human image blob
        "layers": [],  // Assuming no layers, can be modified as needed
        "composite": null  // Assuming no composite image, can be modified as needed
      },
      garmentImage,  // Garment image
      "Short Sleeve Round Neck T-shirts",  // Description of garment
      true,  // Use auto-generated mask
      true,  // Use auto-crop & resizing
      30,  // Denoising Steps
      42  // Seed
    ]);

    console.log(result.data);
  } catch (error) {
    console.error("Error during prediction:", error);
  }
};

run();
