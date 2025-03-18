import { useState } from 'react';
import axios from 'axios';
import './upload.css';

const Timeline = ({ segments, duration }) => {
  return (
    <div className="timeline-container">
      <div className="timeline">
        {segments.map((segment, index) => {
          const [start, end, probability, word] = segment;
          const startPercent = (start / duration) * 100;
          const widthPercent = ((end - start) / duration) * 100;
          
          return (
            <div
              key={index}
              className="censored-segment"
              style={{
                left: `${startPercent}%`,
                width: `${widthPercent}%`
              }}
              title={`${word} (${start.toFixed(2)}s - ${end.toFixed(2)}s)`}
            />
          );
        })}
      </div>
      <div className="timeline-labels">
        <span>0:00</span>
        <span>{Math.floor(duration / 60)}:{String(Math.floor(duration % 60)).padStart(2, '0')}</span>
      </div>
    </div>
  );
};

const Upload = () => {
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [censoredUrl, setCensoredUrl] = useState(null);
  const [processingStatus, setProcessingStatus] = useState('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [censoredSegments, setCensoredSegments] = useState([]);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      // Check if file is audio or video
      if (selectedFile.type.startsWith('audio/') || selectedFile.type.startsWith('video/')) {
        setFile(selectedFile);
        setError(null);
        setCensoredUrl(null); // Reset previous result
        setProcessingStatus('');
      } else {
        setError('Please upload only audio or video files');
        setFile(null);
      }
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }

    setIsUploading(true);
    setProgress(0);
    setError(null);
    setCensoredUrl(null);
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      setProcessingStatus('Uploading file...');
      const response = await axios.post('http://localhost:8080/api/censor', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          setProgress(percentCompleted);
          if (percentCompleted === 100) {
            setProcessingStatus('Processing audio...');
          }
        },
      });

      setProcessingStatus('Processing complete!');
      // Construct full URL for download
      const downloadUrl = `http://localhost:8080${response.data.censoredUrl}`;
      setCensoredUrl(downloadUrl);
      setCensoredSegments(response.data.censoredSegments || []);
    } catch (err) {
      setError('Error processing file: ' + (err.response?.data?.message || err.message));
      setProcessingStatus('');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="upload-container">
      <h1>Thai Profanity Censor</h1>
      <p className="upload-description">
        Upload an audio or video file to automatically detect and censor Thai profanity.
      </p>
      
      <div className="upload-box">
        <input
          type="file"
          accept="audio/*,video/*"
          onChange={handleFileChange}
          className="file-input"
          id="file-input"
        />
        <label htmlFor="file-input" className="file-label">
          {file ? file.name : 'Choose Audio/Video File'}
        </label>
        
        {file && (
          <div className="file-info">
            <p>Selected file: {file.name}</p>
            <p>Size: {(file.size / (1024 * 1024)).toFixed(2)} MB</p>
          </div>
        )}

        <button 
          onClick={handleUpload} 
          disabled={!file || isUploading}
          className="upload-button"
        >
          {isUploading ? 'Processing...' : 'Upload & Censor'}
        </button>
      </div>

      {(isUploading || processingStatus) && (
        <div className="status-container">
          {isUploading && (
            <div className="progress-bar">
              <div className="progress" style={{ width: `${progress}%` }}>
                {progress}%
              </div>
            </div>
          )}
          {processingStatus && (
            <p className="processing-status">{processingStatus}</p>
          )}
        </div>
      )}

      {error && <div className="error-message">{error}</div>}

      {censoredUrl && (
        <div className="result-container">
          <h2>Censored File Ready!</h2>
          <audio
            controls
            className="audio-player"
            onPlay={() => setIsPlaying(true)}
            onPause={() => setIsPlaying(false)}
          >
            <source src={censoredUrl} type={file?.type || 'audio/mpeg'} />
            Your browser does not support the audio element.
          </audio>
          
          {censoredSegments.length > 0 && (
            <div className="censored-info">
              <h3>Censored Segments</h3>
              <Timeline 
                segments={censoredSegments} 
                duration={document.querySelector('audio')?.duration || 0} 
              />
              <div className="censored-list">
                {censoredSegments.map(([start, end, prob, word], index) => (
                  <div key={index} className="censored-item">
                    <span className="censored-word">{word}</span>
                    <span className="censored-time">
                      {start.toFixed(2)}s - {end.toFixed(2)}s
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          <a href={censoredUrl} download className="download-button">
            Download Censored File
          </a>
        </div>
      )}
    </div>
  );
};

export default Upload;
