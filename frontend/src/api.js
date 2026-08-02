const getApiBaseUrl = () => {
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL;
  }
  if (typeof window !== 'undefined') {
    const hostname = window.location.hostname;
    const protocol = window.location.protocol;
    const port = window.location.port;

    // If running Vite dev server (port 3000) or accessing via IP from mobile phone
    if (port === '3000') {
      return `${protocol}//${hostname}:8000`;
    }
    // Production Nginx proxy or relative path
    return '';
  }
  return 'http://localhost:8000';
};

export async function analyzeTableImage(file, options = {}) {
  const formData = new FormData();
  formData.append('image', file);

  if (options.manual_cue_x !== undefined && options.manual_cue_y !== undefined) {
    formData.append('manual_cue_x', options.manual_cue_x);
    formData.append('manual_cue_y', options.manual_cue_y);
  }
  if (options.manual_cue_ball_id) {
    formData.append('manual_cue_ball_id', options.manual_cue_ball_id);
  }
  if (options.felt_color) {
    formData.append('felt_color', options.felt_color);
  }


  const baseUrl = getApiBaseUrl();
  const endpoint = baseUrl ? `${baseUrl}/analyze` : '/analyze';

  let response;
  try {
    response = await fetch(endpoint, {
      method: 'POST',
      body: formData,
    });
  } catch (err) {
    // If direct port 8000 fails, attempt relative path fallback
    if (baseUrl) {
      try {
        response = await fetch('/analyze', {
          method: 'POST',
          body: formData,
        });
      } catch (fallbackErr) {
        throw new Error(`Failed to connect to backend API server (${endpoint}). Please ensure backend is running.`);
      }
    } else {
      throw new Error(`Failed to connect to backend API server (${endpoint}). Please ensure backend is running.`);
    }
  }

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Server error (${response.status}): ${response.statusText}`);
  }

  return await response.json();
}
