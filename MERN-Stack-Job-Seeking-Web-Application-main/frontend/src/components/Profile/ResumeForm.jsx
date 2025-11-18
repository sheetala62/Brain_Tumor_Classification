import React, { useState } from "react";
import ResumePreview from "./ResumePreview";
import "./Profile.css";

const ResumeForm = ({ onResumeChange }) => {
  const [skills, setSkills] = useState("");
  const [education, setEducation] = useState("");
  const [experience, setExperience] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    const resumeData = {
      skills: skills.split(",").map(s => s.trim()),
      education: education.split(";").map(ed => {
        const [degree, institution, year] = ed.split("|");
        return { degree, institution, year };
      }),
      experience: experience.split(";").map(ex => {
        const [role, company, duration, description] = ex.split("|");
        return { role, company, duration, description };
      }),
    };
    onResumeChange(resumeData);
  };

  return (
    <div className="resumeBuilder">
      <h3>Build Your Resume</h3>
      <form onSubmit={handleSubmit} className="profileForm">
        <label>Skills (comma separated)</label>
        <input value={skills} onChange={(e) => setSkills(e.target.value)} />

        <label>Education (Degree|Institution|Year ; separate multiple with ;) </label>
        <input value={education} onChange={(e) => setEducation(e.target.value)} />

        <label>Experience (Role|Company|Duration|Description ; separate multiple with ;) </label>
        <input value={experience} onChange={(e) => setExperience(e.target.value)} />

        <button type="submit">Preview Resume</button>
      </form>
    </div>
  );
};

export default ResumeForm;
