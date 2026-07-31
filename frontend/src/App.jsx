import React, { useState } from 'react';
import UploadScreen from './components/UploadScreen';
import ProcessingScreen from './components/ProcessingScreen';
import ResultScreen from './components/ResultScreen';
import ErrorScreen from './components/ErrorScreen';
import { analyzeTableImage } from './api';
import { Target } from 'lucide-react';

export default function App() {
  const [step, setStep] = useState('upload'); // 'upload' | 'processing' | 'result' | 'error'
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  const handleFileSelected = async (file) => {
    setStep('processing');
    setErrorMsg(null);

    try {
      const data = await analyzeTableImage(file);
      const hasDirect = data.direct_shots && data.direct_shots.length > 0;
      const hasBank = data.bank_shots && data.bank_shots.length > 0;
      const hasKick = data.kick_shots && data.kick_shots.length > 0;

      if (!hasDirect && !hasBank && !hasKick) {
        setErrorMsg('No valid shots found — every possible bank is blocked or misses all pockets.');
        setStep('error');
      } else {
        setResult(data);
        setStep('result');
      }
    } catch (err) {
      setErrorMsg(err.message || 'Failed to analyze pool table image.');
      setStep('error');
    }
  };

  const handleReset = () => {
    setStep('upload');
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
        <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>v0.6.0</span>
      </header>

      <main className="container">
        {step === 'upload' && <UploadScreen onFileSelected={handleFileSelected} error={errorMsg} />}
        {step === 'processing' && <ProcessingScreen />}
        {step === 'result' && result && <ResultScreen result={result} onReset={handleReset} />}
        {step === 'error' && <ErrorScreen message={errorMsg} onReset={handleReset} />}
      </main>
    </>
  );
}
