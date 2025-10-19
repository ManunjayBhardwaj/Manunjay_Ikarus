import React, {useState} from 'react'
import axios from 'axios'

export default function ChatPage(){
  const [messages, setMessages] = useState([{
    role: 'assistant',
    content: 'Hi! I can help you pick the perfect furniture for your space. Tell me about your room size, priorities (comfort, storage, style), preferred materials, and budget.'
  }])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [temperature, setTemperature] = useState(0.3)

  async function send(){
    if(!input.trim()) return
    const userMsg = {role:'user', content: input}
    const newMsgs = messages.concat([userMsg])
    setMessages(newMsgs)
    setInput('')
    setLoading(true)
    try{
      const resp = await axios.post('/api/chat', {messages: newMsgs, preferences: {temperature}})
      const data = resp.data
      const assistant = {role:'assistant', content: `Suggestion: ${data.suggestion}\n\n${data.description}`}
      setMessages(m => m.concat([assistant]))
    }catch(e){
      setMessages(m => m.concat([{role:'assistant', content: 'Sorry, something went wrong.'}]))
    }finally{
      setLoading(false)
    }
  }

  return (
    <div style={{padding:20}}>
      <h2 style={{marginTop:0}}>Furniture Recommender ChatBot</h2>
      <div style={{height:400, overflow:'auto', border:'1px solid rgba(0,0,0,0.06)', padding:12, borderRadius:8, marginBottom:12, background:'var(--card)'}}>
        {messages.map((m,idx)=> (
          <div key={idx} style={{marginBottom:12}}>
            <div style={{fontSize:12, color:'var(--muted)', textTransform:'capitalize'}}>{m.role}</div>
            <div style={{whiteSpace:'pre-wrap', marginTop:6, color:'var(--text)'}}>{m.content}</div>
          </div>
        ))}
      </div>

      <div style={{display:'flex', gap:8, alignItems:'center'}}>
        <input value={input} onChange={e=>setInput(e.target.value)} style={{flex:1, padding:8, borderRadius:6, border:'1px solid rgba(0,0,0,0.06)', background:'transparent', color:'var(--text)'}} placeholder="Type your requirements..." />
        <div style={{display:'flex', alignItems:'center', gap:8, marginRight:8}}>
          <label style={{fontSize:12,color:'var(--muted)'}}>Temp</label>
          <input type="range" min="0" max="1" step="0.1" value={temperature} onChange={e=>setTemperature(Number(e.target.value))} />
        </div>
        <button onClick={send} disabled={loading} style={{padding:'8px 12px', borderRadius:6}}>{loading? '...' : 'Send'}</button>
      </div>
    </div>
  )
}
