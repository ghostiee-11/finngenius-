// server.js
const express = require('express');
const dotenv = require('dotenv');
const cors = require('cors');
const connectDB = require('./config/db');
const authRoutes = require('./routes/authRoutes');

// Load env vars
dotenv.config();

// Connect to database
connectDB();

const app = express();

// --- Middleware ---

// CORS - Configure allowed origins (IMPORTANT!)
// Update the origin to match where your frontend is served from
const corsOptions = {
  origin: 'http://127.0.0.1:3000', // Allow requests from frontend on port 3000
  optionsSuccessStatus: 200
};
app.use(cors(corsOptions));

// Body Parser - To accept JSON data in request body
app.use(express.json());

// --- Mount Routers ---
app.use('/api/auth', authRoutes);
// Mount other routers here (e.g., for data/news if they are separate)

// --- Basic Root Route (Optional) ---
app.get('/', (req, res) => {
  res.send('FinGenius API Running');
});

// --- Basic Error Handler ---
app.use((err, req, res, next) => {
    console.error("Unhandled Error:", err.stack);
    res.status(err.status || 500).json({
        success: false,
        message: err.message || 'Internal Server Error'
    });
});

const PORT = process.env.PORT || 5015; // Backend still runs on 5015

const server = app.listen(
  PORT,
  console.log(`Server running in ${process.env.NODE_ENV || 'development'} mode on port ${PORT}`)
);

// Handle unhandled promise rejections
process.on('unhandledRejection', (err, promise) => {
  console.error(`Unhandled Rejection: ${err.message}`);
  server.close(() => process.exit(1));
});