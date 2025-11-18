// 1. Load environment variables first
import { config } from "dotenv";
config({ path: "./config/config.env" }); // MUST be at the very top

// 2. Import the rest
import app from "./app.js";
import cloudinary from "cloudinary";

// 3. Configure Cloudinary
cloudinary.v2.config({
  cloud_name: process.env.CLOUDINARY_CLIENT_NAME,
  api_key: process.env.CLOUDINARY_CLIENT_API,
  api_secret: process.env.CLOUDINARY_CLIENT_SECRET,
});

// 4. Start the server
app.listen(process.env.PORT, () => {
  console.log(`Server running at port ${process.env.PORT}`);
});
