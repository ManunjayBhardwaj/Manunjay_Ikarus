Netlify deployment notes

Set the following environment variable in your Netlify site settings (Build & deploy > Environment):

- REACT_APP_API_URL — the full URL of your backend API (for example: https://api.example.com). If not set, the app defaults to http://localhost:8000.

When deploying, Netlify will run the build defined in `netlify.toml` which installs dependencies in the `frontend` folder and runs `npm run build` there.
