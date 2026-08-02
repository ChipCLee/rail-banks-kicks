import React, { useState } from 'react';
import UploadScreen from './components/UploadScreen';
import ProcessingScreen from './components/ProcessingScreen';
import ResultScreen from './components/ResultScreen';
import TeachModeScreen from './components/TeachModeScreen';
import ErrorScreen from './components/ErrorScreen';
import { analyzeTableImage } from './api';
import { Target } from 'lucide-react';

export default function App() {
  const [step, setStep] = useState('upload'); // 'upload' | 'processing' | 'teach' | 'result' | 'error'
  const [currentFile, setCurrentFile] = useState(null);
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  const processImage = async (file, options = {}) => {
    setStep('processing');
    setErrorMsg(null);

    try {
      const data = await analyzeTableImage(file, options);
      setResult(data);

      if (data.cue_detected === false) {
        setStep('teach');
        return;
      }

      setStep('result');
    } catch (err) {
      setErrorMsg(err.message || 'Failed to analyze pool table image.');
      setStep('error');
    }
  };

  const handleFileSelected = (file) => {
    setCurrentFile(file);
    processImage(file);
  };

  const handleSetCueBallLocation = (options) => {
    if (currentFile) {
      processImage(currentFile, options);
    }
  };

  const handleReset = () => {
    setStep('upload');
    setCurrentFile(null);
    setResult(null);
    setErrorMsg(null);
  };

  return (
    <>
      <header className="header">
        <div className="logo">
          <Target size={24} />
          <span>Rail-Kick</span>
        </div>
        <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>v0.7.0</span>
      </header>

      <main className="container">
        {step === 'upload' && (
          <UploadScreen 
            onFileSelected={handleFileSelected} 
            error={errorMsg} 
          />
        )}
        {step === 'processing' && <ProcessingScreen />}
        {step === 'teach' && result && (
          <TeachModeScreen 
            result={result} 
            onSetCueBallLocation={handleSetCueBallLocation}
            onReset={handleReset}
          />
        )}
        {step === 'result' && result && (
          <ResultScreen 
            result={result} 
            onReset={handleReset} 
            onEnterTeachMode={() => setStep('teach')}
          />
        )}
        {step === 'error' && <ErrorScreen message={errorMsg} onReset={handleReset} />}
      </main>
    </>
  );
}
