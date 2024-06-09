import express from "express";
import multer from "multer";
import axios from "axios";
import sharp from "sharp";
import { client } from "@gradio/client";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { HfInference } from "@huggingface/inference";
import dotenv from "dotenv";

// Load environment variables from .env file
dotenv.config();

// Derive __dirname from import.meta.url
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = 3000;

const upload = multer({ dest: "uploads/" });
const hf = new HfInference(process.env.HUGGING_FACE_API_KEY);

app.use(express.json());

const readLocalFile = (filePath) => {
  return new Promise((resolve, reject) => {
    fs.readFile(filePath, (err, data) => {
      if (err) {
        reject(err);
      } else {
        resolve(data);
      }
    });
  });
};

const downloadImage = async (url, filepath) => {
  const response = await axios({
    url,
    responseType: "stream",
  });
  return new Promise((resolve, reject) => {
    const writer = fs.createWriteStream(filepath);
    response.data.pipe(writer);
    writer.on("finish", () => resolve());
    writer.on("error", reject);
  });
};

const resizeImage = async (inputPath, outputPath, width, height) => {
  await sharp(inputPath)
    .resize(width, height)
    .toFile(outputPath);
};

app.post("/tryon", upload.fields([{ name: "humanImage" }, { name: "garmentImage" }]), async (req, res) => {
  try {
    const appClient = await client("alexff91/FitMirror");

    let humanImagePath;
    let garmentImagePath;
    const humanImageResizedPath = path.join(__dirname, "uploads", "humanImage_resized.jpg");
    const garmentImageResizedPath = path.join(__dirname, "uploads", "garmentImage_resized.jpg");

    if (req.files && req.files.humanImage) {
      humanImagePath = req.files.humanImage[0].path;
    } else if (req.body.humanImageURL) {
      humanImagePath = path.join(__dirname, "uploads", "humanImage.jpg");
      await downloadImage(req.body.humanImageURL, humanImagePath);
    }

    if (req.files && req.files.garmentImage) {
      garmentImagePath = req.files.garmentImage[0].path;
    } else if (req.body.garmentImageURL) {
      garmentImagePath = path.join(__dirname, "uploads", "garmentImage.jpg");
      await downloadImage(req.body.garmentImageURL, garmentImagePath);
    }

    // Resize images
    await resizeImage(humanImagePath, humanImageResizedPath, 400, 600);
    await resizeImage(garmentImagePath, garmentImageResizedPath, 400, 600);

    const humanImage = await readLocalFile(humanImageResizedPath);
    const garmentImage = await readLocalFile(garmentImageResizedPath);

    // Use provided description or default description
    const garmentDescription = req.body.garmentDescription || "cloth fitting the person shape";

    const result = await appClient.predict("/tryon", [
      {
        "background": humanImage,
        "layers": [],
        "composite": null
      },
      garmentImage,
      garmentDescription,  // Description of garment
      true,  // Use auto-generated mask
      true,  // Use auto-crop & resizing
      30,  // Denoising Steps
      42  // Seed
    ]);

    res.json({ result: result.data });
  } catch (error) {
    console.error("Error during prediction:", error);
    res.status(500).json({ error: error.message });
  }
});

app.listen(port, () => {
  console.log(`Try-on service listening at http://localhost:${port}`);
});
