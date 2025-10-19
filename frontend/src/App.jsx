import React, {useState, useEffect} from 'react'
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom'
import './styles/ui.css'
import RecommendPage from './pages/RecommendPage'
import AnalyticsPage from './pages/AnalyticsPage'
import ChatPage from './pages/ChatPage'

export default function App(){
  const [dark, setDark] = useState(()=> localStorage.getItem('ikarus:dark') === '1')
  useEffect(()=>{
    document.documentElement.classList.toggle('dark', dark)
    localStorage.setItem('ikarus:dark', dark? '1':'0')
  },[dark])

  return (
    <BrowserRouter>
      <header className="app-shell">
        <nav className="container">
          <div className="brand"><div style={{width:36,height:36,background:'linear-gradient(135deg,#7c3aed,#06b6d4)',borderRadius:8}}/> Ikarus</div>
          <div className="nav-links">
            <Link to="/">Recommend</Link>
            <Link to="/analytics">Analytics</Link>
            <Link to="/chat">Chat</Link>
            <button aria-label="Toggle dark mode" onClick={()=>setDark(d=>!d)} className="chip">{dark? '☀️':'🌙'}</button>
          </div>
        </nav>
      </header>

      <main className="container" style={{paddingTop:12}}>
        <Routes>
          <Route path="/" element={<RecommendPage/>} />
          <Route path="/analytics" element={<AnalyticsPage/>} />
          <Route path="/chat" element={<ChatPage/>} />
        </Routes>
      </main>

  <footer className="footer">Manunjay Bhardwaj</footer>
    </BrowserRouter>
  )
}
