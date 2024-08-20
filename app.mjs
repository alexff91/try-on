import express from 'express';
import multer from 'multer';
import axios from 'axios';
import sharp from 'sharp';
import { client } from '@gradio/client';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import rateLimit from 'express-rate-limit';

// Load environment variables from .env file
dotenv.config();

// Derive __filename and __dirname from import.meta.url
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = 8080;
const host = '0.0.0.0';

const upload = multer({ dest: "uploads/" });

// Increase the payload limit
app.use(express.json({ limit: '50mb' }));

import { HfInference } from '@huggingface/inference';

const hf = new HfInference(process.env.HUGGING_FACE_API_KEY); // Make sure the API key is available

// Helper functions with logs
const readLocalFile = (filePath) => {
  console.log(`Reading local file: ${filePath}`);
  return new Promise((resolve, reject) => {
    fs.readFile(filePath, (err, data) => {
      if (err) {
        console.error(`Error reading file: ${filePath}`, err);
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
  console.log(`Converting buffer to base64`);
  return buffer.toString('base64');
};

const base64ToBuffer = (base64) => {
  console.log(`Converting base64 to buffer`);
  return Buffer.from(base64, 'base64');
};

// Rate limiting middleware
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // Limit each IP to 100 requests per windowMs
  message: "Too many requests from this IP, please try again later."
});

app.use(limiter);

const getModelUrlByType = (type) => {
  console.log(`Getting model URL for type: ${type}`);
  switch (type) {
    case 'up':
      return "alexff91/FitMirrorUp";
    case 'down':
      return "alexff91/FitMirror-Down";
    case 'dress':
      return "alexff91/FitMirror-Dress";
    default:
      console.error("Invalid type parameter. Allowed values are 'up', 'down', or 'dress'.");
      throw new Error("Invalid type parameter. Allowed values are 'up', 'down', or 'dress'.");
  }
};

// Try-on (File Upload) with model selection based on type
app.post("/tryon", upload.fields([{ name: "humanImage" }, { name: "garmentImage" }]), async (req, res) => {
  try {
    console.log("Received /tryon request");

    const { type } = req.body;
    const modelUrl = getModelUrlByType(type);
    const appClient = await client(modelUrl);

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

    const garmentDescription = req.body.garmentDescription || "cloth fitting the person shape";

    console.log(`Sending prediction request to the ${modelUrl} model`);
    const result = await appClient.predict("/tryon", [
      {
        "background": humanImageBuffer,
        "layers": [],
        "composite": null
      },
      garmentImageBuffer,
      garmentDescription,
      true,
      true,
      30,
      42
    ]);

    console.log("Prediction completed successfully", result);
    res.json({ result: result.data });
  } catch (error) {
    console.error("Error during prediction:", error.response ? error.response.data : error.message);
    res.status(500).json({ error: error.response ? error.response.data : error.message });
  }
});

// Try-on (Base64) with model selection based on type
app.post("/tryon/base64", async (req, res) => {
  try {
    console.log("Received /tryon/base64 request");

    const { type, humanImageBase64, garmentImageBase64, garmentDescription = "cloth fitting the person shape" } = req.body;
    const modelUrl = getModelUrlByType(type);
    const appClient = await client(modelUrl);

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

    console.log(`Sending prediction request to the ${modelUrl} model`);
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

// Try-on (Media) with model selection based on type
app.post("/tryon/media", upload.fields([{ name: "humanImage" }, { name: "garmentImage" }]), async (req, res) => {
  try {
    console.log("Received /tryon/media request");

    const { type } = req.body;
    const modelUrl = getModelUrlByType(type);
    const appClient = await client(modelUrl);

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

    const garmentDescription = req.body.garmentDescription || "cloth fitting the person shape";

    console.log(`Sending prediction request to the ${modelUrl} model`);
    const result = await appClient.predict("/tryon", [
      {
        "background": humanImageBuffer,
        "layers": [],
        "composite": null
      },
      garmentImageBuffer,
      garmentDescription,
      true,
      true,
      30,
      42
    ]);

    const outputUrl = result.data[0].url;
    const outputImageResponse = await axios.get(outputUrl, { responseType: 'arraybuffer' });

    res.setHeader('Content-Type', 'image/png');
    res.send(outputImageResponse.data);
  } catch (error) {
    console.error("Error during prediction:", error.response ? error.response.data : error.message);
    res.status(500).json({ error: error.response ? error.response.data : error.message });
  }
});

// New endpoint accepting 2 URLs and text, and returning media
app.post("/tryon/url", async (req, res) => {
  try {
    console.log("Received /tryon/url request");

    const { type, humanImageURL, garmentImageURL, garmentDescription = "cloth fitting the person shape" } = req.body;
    const modelUrl = getModelUrlByType(type);
    const appClient = await client(modelUrl);

    if (!humanImageURL || !garmentImageURL) {
      return res.status(400).json({ error: "Both humanImageURL and garmentImageURL are required." });
    }

    const humanImagePath = path.join(__dirname, "uploads", "humanImage_from_url.jpg");
    const garmentImagePath = path.join(__dirname, "uploads", "garmentImage_from_url.jpg");

    await downloadImage(humanImageURL, humanImagePath);
    await downloadImage(garmentImageURL, garmentImagePath);

    const humanImageResizedPath = path.join(__dirname, "uploads", "humanImage_resized.jpg");
    const garmentImageResizedPath = path.join(__dirname, "uploads", "garmentImage_resized.jpg");

    await resizeImage(humanImagePath, humanImageResizedPath, 400, 600);
    await resizeImage(garmentImagePath, garmentImageResizedPath, 400, 600);

    const resizedHumanImageBuffer = await readLocalFile(humanImageResizedPath);
    const resizedGarmentImageBuffer = await readLocalFile(garmentImageResizedPath);

    console.log(`Sending prediction request to the ${modelUrl} model`);
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

    res.setHeader('Content-Type', 'image/png');
    res.send(outputImageResponse.data);
  } catch (error) {
    console.error("Error during prediction:", error.response ? error.response.data : error.message);
    res.status(500).json({ error: error.response ? error.response.data : error.message });
  }
});

// New API 2: Check Wardrobe
app.post("/check-wardrobe", upload.single("garmentImage"), async (req, res) => {
  try {
    console.log("Received /check-wardrobe request");

    if (!req.file) {
      return res.status(400).json({ error: "Garment image is required." });
    }

    // Resize the image to the required size for the model (224x224 in this case)
    const processedImage = await sharp(req.file.path).resize(224, 224).toBuffer();

    // Perform classification using the Hugging Face Inference API
    const result = await hf.image_classification({
      model: "microsoft/beit-base-patch16-224-pt22k-ft22k",
      data: processedImage,
    });

    console.log("Wardrobe check completed successfully", result);
    res.json(result);
  } catch (error) {
    console.error("Error during wardrobe check:", error.message);
    res.status(500).json({ error: error.message });
  }
});

// New API 3: Check if Image Contains a Person
app.post("/check-person", upload.single("image"), async (req, res) => {
  try {
    console.log("Received /check-person request");

    if (!req.file) {
      return res.status(400).json({ error: "Image is required." });
    }

    // Resize the image to the required size for the model (512x512 in this case)
    const processedImage = await sharp(req.file.path).resize(512, 512).toBuffer();

    // Perform segmentation using the Hugging Face Inference API
    const result = await hf.semantic_segmentation({
      model: "nvidia/segformer-b0-finetuned-ade-512-512",
      data: processedImage,
    });

    console.log("Person check completed successfully", result);
    res.json(result);
  } catch (error) {
    console.error("Error during person check:", error.message);
    res.status(500).json({ error: error.message });
  }
});

app.listen(port, host, () => {
  console.log(`Try-on service listening at http://${host}:${port}`);
});
