import { catchAsyncErrors } from "../middlewares/catchAsyncError.js";
import { User } from "../models/userSchema.js";
import ErrorHandler from "../middlewares/error.js";
import { sendToken } from "../utils/jwtToken.js";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import PDFDocument from "pdfkit";

// ------------------ Auth Controllers ------------------

export const register = catchAsyncErrors(async (req, res, next) => {
  const { name, email, phone, password, role } = req.body;
  if (!name || !email || !phone || !password || !role) {
    return next(new ErrorHandler("Please fill full form!"));
  }
  const isEmail = await User.findOne({ email });
  if (isEmail) {
    return next(new ErrorHandler("Email already registered!"));
  }
  const user = await User.create({
    name,
    email,
    phone,
    password,
    role,
  });
  sendToken(user, 201, res, "User Registered!");
});

export const login = catchAsyncErrors(async (req, res, next) => {
  const { email, password, role } = req.body;
  if (!email || !password || !role) {
    return next(new ErrorHandler("Please provide email ,password and role."));
  }
  const user = await User.findOne({ email }).select("+password");
  if (!user) {
    return next(new ErrorHandler("Invalid Email Or Password.", 400));
  }
  const isPasswordMatched = await user.comparePassword(password);
  if (!isPasswordMatched) {
    return next(new ErrorHandler("Invalid Email Or Password.", 400));
  }
  if (user.role !== role) {
    return next(
      new ErrorHandler(`User with provided email and ${role} not found!`, 404)
    );
  }
  sendToken(user, 201, res, "User Logged In!");
});

export const logout = catchAsyncErrors(async (req, res, next) => {
  res
    .status(201)
    .cookie("token", "", {
      httpOnly: true,
      expires: new Date(Date.now()),
    })
    .json({
      success: true,
      message: "Logged Out Successfully.",
    });
});

export const getUser = catchAsyncErrors((req, res, next) => {
  const user = req.user;
  res.status(200).json({
    success: true,
    user,
  });
});

// ------------------ Resume Controllers ------------------

// Upload an existing resume (PDF)
export const uploadResume = catchAsyncErrors(async (req, res, next) => {
  if (!req.file) {
    return next(new ErrorHandler("Please upload a resume file", 400));
  }

  const user = await User.findById(req.user.id);
  user.resume = req.file.path; // save resume path in DB
  await user.save();

  res.status(200).json({
    success: true,
    message: "Resume uploaded successfully",
    resumePath: user.resume,
  });
});

// Auto-generate resume PDF from user details
export const generateResume = catchAsyncErrors(async (req, res, next) => {
  const user = await User.findById(req.user.id);
  if (!user) {
    return next(new ErrorHandler("User not found", 404));
  }

  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);
  const resumePath = path.join(__dirname, `../uploads/resume-${user._id}.pdf`);

  const doc = new PDFDocument();
  const stream = fs.createWriteStream(resumePath);
  doc.pipe(stream);

  // Resume Content Example
  doc.fontSize(20).text(`${user.name}`, { align: "center" });
  doc.fontSize(14).text(`Email: ${user.email}`);
  doc.fontSize(14).text(`Phone: ${user.phone}`);
  doc.moveDown();

  doc.fontSize(16).text("Profile");
  doc.fontSize(12).text(`Role: ${user.role}`);
  // later expand with education, skills, experience from req.body

  doc.end();

  stream.on("finish", async () => {
    user.resume = resumePath;
    await user.save();

    res.status(200).json({
      success: true,
      message: "Resume generated successfully",
      resumePath: user.resume,
    });
  });
});
