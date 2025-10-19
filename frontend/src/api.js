import axios from 'axios';

const client = axios.create({ baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000' });

export async function recommend(query, k=5) {
  const res = await client.post('/api/recommend', { query, k });
  return res.data;
}

export async function getAnalytics() {
  const res = await client.get('/api/analytics');
  return res.data;
}

export async function ingest() {
  const res = await client.post('/api/ingest');
  return res.data;
}
