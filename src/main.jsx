import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import Chapter08 from './Chapter08.jsx'


createRoot(document.getElementById('root')).render(
  <StrictMode>
    <Chapter08 />
  </StrictMode>,
)
