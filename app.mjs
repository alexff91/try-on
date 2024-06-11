import express from 'express';
import multer from 'multer';
import axios from 'axios';
import sharp from 'sharp';
import { client } from '@gradio/client';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { HfInference } from '@huggingface/inference';
import dotenv from 'dotenv';

// Load environment variables from .env file
dotenv.config();

// Derive __filename and __dirname from import.meta.url
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = 8080;
const host = '0.0.0.0';

const upload = multer({ dest: "uploads/" });
const hf = new HfInference(process.env.HUGGING_FACE_API_KEY);

// Increase the payload limit
app.use(express.json({ limit: '50mb' })); // Increase the limit to 50mb or as needed

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
  console.log(`Downloading image from ${url} to ${filepath}`);
  const response = await axios({
    url,
    responseType: "stream",
    timeout: 60000 // 60 seconds timeout
  });
  return new Promise((resolve, reject) => {
    const writer = fs.createWriteStream(filepath);
    response.data.pipe(writer);
    writer.on("finish", () => {
      console.log(`Image downloaded to ${filepath}`);
      resolve();
    });
    writer.on("error", (err) => {
      console.error(`Error downloading image: ${err.message}`);
      reject(err);
    });
  });
};

const resizeImage = async (inputPath, outputPath, width, height) => {
  console.log(`Resizing image ${inputPath} to ${width}x${height} at ${outputPath}`);
  await sharp(inputPath)
    .resize(width, height)
    .toFile(outputPath);
  console.log(`Image resized to ${outputPath}`);
};

const bufferToBase64 = (buffer) => {
  return buffer.toString('base64');
};

const base64ToBuffer = (base64) => {
  return Buffer.from(base64, 'base64');
};

app.post("/tryon", upload.fields([{ name: "humanImage" }, { name: "garmentImage" }]), async (req, res) => {
  try {
    console.log("Received /tryon request");
    const appClient = await client("alexff91/FitMirror");

    let humanImagePath;
    let garmentImagePath;
    const humanImageResizedPath = path.join(__dirname, "uploads", "humanImage_resized.jpg");
    const garmentImageResizedPath = path.join(__dirname, "uploads", "garmentImage_resized.jpg");

    if (req.files && req.files.humanImage) {
      humanImagePath = req.files.humanImage[0].path;
      console.log(`Human image uploaded: ${humanImagePath}`);
    } else if (req.body.humanImageURL) {
      humanImagePath = path.join(__dirname, "uploads", "humanImage.jpg");
      await downloadImage(req.body.humanImageURL, humanImagePath);
    }

    if (req.files && req.files.garmentImage) {
      garmentImagePath = req.files.garmentImage[0].path;
      console.log(`Garment image uploaded: ${garmentImagePath}`);
    } else if (req.body.garmentImageURL) {
      garmentImagePath = path.join(__dirname, "uploads", "garmentImage.jpg");
      await downloadImage(req.body.garmentImageURL, garmentImagePath);
    }

    // Resize images
    await resizeImage(humanImagePath, humanImageResizedPath, 400, 600);
    await resizeImage(garmentImagePath, garmentImageResizedPath, 400, 600);

    const humanImageBuffer = await readLocalFile(humanImageResizedPath);
    const garmentImageBuffer = await readLocalFile(garmentImageResizedPath);

    // Use provided description or default description
    const garmentDescription = req.body.garmentDescription || "cloth fitting the person shape";

    console.log("Sending prediction request to the model");
    const result = await appClient.predict("/tryon", [
      {
        "background": humanImageBuffer,
        "layers": [],
        "composite": null
      },
      garmentImageBuffer,
      garmentDescription,  // Description of garment
      true,  // Use auto-generated mask
      true,  // Use auto-crop & resizing
      30,  // Denoising Steps
      42  // Seed
    ]);

    console.log("Prediction completed successfully", result);
    res.json({ result: result.data });
  } catch (error) {
    console.error("Error during prediction:", error.response ? error.response.data : error.message);
    res.status(500).json({ error: error.response ? error.response.data : error.message });
  }
});

app.post("/tryon/base64", async (req, res) => {
  try {
    console.log("Received /tryon/base64 request");
    const appClient = await client("alexff91/FitMirror");

    const { humanImageBase64, garmentImageBase64, garmentDescription = "cloth fitting the person shape" } = req.body;

    if (!humanImageBase64 || !garmentImageBase64) {
      return res.status(400).json({ error: "Both humanImageBase64 and garmentImageBase64 are required." });
    }

    const humanImageBuffer = base64ToBuffer(humanImageBase64);
    const garmentImageBuffer = base64ToBuffer(garmentImageBase64);

    const humanImagePath = path.join(__dirname, "uploads", "humanImage_from_base64.jpg");
    const garmentImagePath = path.join(__dirname, "uploads", "garmentImage_from_base64.jpg");

    fs.writeFileSync(humanImagePath, humanImageBuffer);
    fs.writeFileSync(garmentImagePath, garmentImageBuffer);

    const humanImageResizedPath = path.join(__dirname, "uploads", "humanImage_resized.jpg");
    const garmentImageResizedPath = path.join(__dirname, "uploads", "garmentImage_resized.jpg");

    await resizeImage(humanImagePath, humanImageResizedPath, 400, 600);
    await resizeImage(garmentImagePath, garmentImageResizedPath, 400, 600);

    const resizedHumanImageBuffer = await readLocalFile(humanImageResizedPath);
    const resizedGarmentImageBuffer = await readLocalFile(garmentImageResizedPath);

    console.log("Sending prediction request to the model");
    const result = await appClient.predict("/tryon", [
      {
        "background": resizedHumanImageBuffer,
        "layers": [],
        "composite": null
      },
      resizedGarmentImageBuffer,
      garmentDescription,
      true,
      true,
      30,
      42
    ]);

    const outputUrl = result.data[0].url;
    const outputImageResponse = await axios.get(outputUrl, { responseType: 'arraybuffer' });
    const outputImageBase64 = Buffer.from(outputImageResponse.data, 'binary').toString('base64');

    console.log("Prediction completed successfully");
    res.json({ result: outputImageBase64 });
  } catch (error) {
    console.error("Error during prediction:", error.response ? error.response.data : error.message);
    res.status(500).json({ error: error.response ? error.response.data : error.message });
  }
});

app.listen(port, host, () => {
  console.log(`Try-on service listening at http://${host}:${port}`);
});
