const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

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

  const response = await fetch(`${API_BASE_URL}/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Server error: ${response.statusText}`);
  }

  return await response.json();
}
