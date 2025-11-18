import React, { useContext, useState, useEffect } from "react";
import axios from "axios";
import toast from "react-hot-toast";
import { Context } from "../../main";
import ResumeForm from "./ResumeForm";
import ResumePreview from "./ResumePreview";
import "./Profile.css";

const Profile = () => {
  const { user, setUser } = useContext(Context);

  // Profile info
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [phone, setPhone] = useState("");
  const [skills, setSkills] = useState([]);
  const [education, setEducation] = useState([]);
  const [experience, setExperience] = useState([]);
  const [resumeFile, setResumeFile] = useState(null);

  // Resume Builder state
  const [resumeData, setResumeData] = useState(null);

  useEffect(() => {
    if (user) {
      setName(user.name || "");
      setEmail(user.email || "");
      setPhone(user.phone || "");
      setSkills(user.skills || []);
      setEducation(user.education || []);
      setExperience(user.experience || []);
      if (user.skills || user.education || user.experience) {
        setResumeData({
          skills: user.skills || [],
          education: user.education || [],
          experience: user.experience || [],
        });
      }
    }
  }, [user]);

  // Handle resume PDF upload
  const handleResumeUpload = async (e) => {
    e.preventDefault();
    if (!resumeFile) return toast.error("Please select a resume file!");
    const formData = new FormData();
    formData.append("resume", resumeFile);

    try {
      const { data } = await axios.post(
        "http://localhost:5000/api/v1/user/upload-resume",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
          withCredentials: true,
        }
      );
      toast.success(data.message);
      setUser((prev) => ({ ...prev, resumeUrl: data.resumeUrl }));
    } catch (error) {
      toast.error(error.response?.data?.message || "Upload failed!");
    }
  };

  // Update profile including resume builder data
  const handleProfileUpdate = async (e) => {
    e.preventDefault();
    try {
      const { data } = await axios.put(
        "http://localhost:5000/api/v1/user/update",
        { name, phone, skills: resumeData?.skills, education: resumeData?.education, experience: resumeData?.experience },
        { withCredentials: true }
      );
      toast.success(data.message);
      setUser(data.user);
    } catch (error) {
      toast.error(error.response?.data?.message || "Update failed!");
    }
  };

  return (
    <section className="profilePage">
      <div className="profileContainer">
        <h2>My Profile</h2>

        <form className="profileForm" onSubmit={handleProfileUpdate}>
          <label>Name</label>
          <input value={name} onChange={(e) => setName(e.target.value)} />

          <label>Email (read-only)</label>
          <input value={email} readOnly />

          <label>Phone</label>
          <input value={phone} onChange={(e) => setPhone(e.target.value)} />

          <button type="submit">Update Profile</button>
        </form>

        <h3>Resume Builder</h3>
        <ResumeForm onResumeChange={setResumeData} />
        <ResumePreview resumeData={resumeData} />

        <h3>Upload Resume PDF</h3>
        <form className="resumeForm" onSubmit={handleResumeUpload}>
          <input
            type="file"
            accept=".pdf"
            onChange={(e) => setResumeFile(e.target.files[0])}
          />
          <button type="submit">Upload</button>
        </form>

        {user?.resumeUrl && (
          <a href={user.resumeUrl} target="_blank" rel="noopener noreferrer">
            View Uploaded Resume
          </a>
        )}
      </div>
    </section>
  );
};

export default Profile;
