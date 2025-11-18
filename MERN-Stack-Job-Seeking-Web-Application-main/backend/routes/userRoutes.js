import express from "express";
import { login, register, logout, getUser, updateProfile } from "../controllers/userController.js";
import { isAuthenticated } from "../middlewares/auth.js";
import fileUpload from "express-fileupload"; // to handle file uploads

const router = express.Router();

// User Authentication Routes
router.post("/register", register);
router.post("/login", login);
router.get("/logout", isAuthenticated, logout);
router.get("/getuser", isAuthenticated, getUser);

// 🔹 Update Profile / Resume Builder
router.put(
  "/updateprofile",
  isAuthenticated,
  fileUpload({ useTempFiles: true }), // to handle uploaded resume
  updateProfile
);

export default router;
