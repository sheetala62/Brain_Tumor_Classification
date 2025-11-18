import React from "react";
import "./Profile.css";

const ResumePreview = ({ resumeData }) => {
  if (!resumeData) return null;

  const { skills, education, experience } = resumeData;

  return (
    <div className="resumePreview">
      <h3>Resume Preview</h3>
      <section>
        <h4>Skills</h4>
        <ul>
          {skills.map((skill, i) => (
            <li key={i}>{skill}</li>
          ))}
        </ul>
      </section>

      <section>
        <h4>Education</h4>
        {education.map((ed, i) => (
          <div key={i}>
            <strong>{ed.degree}</strong> - {ed.institution} ({ed.year})
          </div>
        ))}
      </section>

      <section>
        <h4>Experience</h4>
        {experience.map((ex, i) => (
          <div key={i}>
            <strong>{ex.role}</strong> - {ex.company} ({ex.duration})
            <p>{ex.description}</p>
          </div>
        ))}
      </section>
    </div>
  );
};

export default ResumePreview;
