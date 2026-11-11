# EventLens Mobile

Expo mobile companion for EventLens. It uses the same Flask API and Supabase Auth as the website.

## Setup

1. Copy `.env.example` to `.env`.
2. Set the Supabase project URL and publishable key used by the website.
3. Set `EXPO_PUBLIC_API_BASE_URL` to the backend API URL.
   - Android emulator: `http://10.0.2.2:5000/api`
   - iOS simulator: `http://127.0.0.1:5000/api`
   - Physical device: use the computer's LAN IP and ensure the phone and computer share a network.
4. Install and start:

   ```bash
   npm install
   npx expo start
   ```

Sign in with an existing EventLens account, choose a video, and run analysis. The backend must have its Supabase auth environment variables configured as well.
