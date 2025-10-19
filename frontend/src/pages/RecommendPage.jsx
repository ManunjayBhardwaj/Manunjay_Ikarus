import React, {useState} from 'react'
import {recommend} from '../api'
import ProductCard from '../components/ProductCard'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import Input from '../components/ui/Input'
import Skeleton from '../components/ui/Skeleton'

const samples = [
  'minimalist wooden chair under ₹5000',
  'comfortable three-seater sofa for living room',
  'compact study desk with drawer'
]

export default function RecommendPage(){
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState([])

  async function onSubmit(e){
    e && e.preventDefault()
    setLoading(true)
    try{
      const items = await recommend(query, 5)
      setResults(items)
    }catch(err){
      console.error(err)
      alert('Error or no backend running')
    }
    setLoading(false)
  }

  return (
    <div className="recommend-dark">
      <section className="card-lg" style={{marginBottom:16}}>
        <h1 className="title">Find furniture you’ll love</h1>
        <p className="subtitle">Describe what you're looking for — style, size, material, or budget.</p>

        <form onSubmit={onSubmit} style={{display:'flex', gap:12, alignItems:'center'}}>
          <div className="pill-input" style={{flex:1}}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden><path d="M21 21l-4.35-4.35" stroke="#9CA3AF" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>
            <Input value={query} onChange={e=>setQuery(e.target.value)} placeholder="e.g. minimalist wooden chair under ₹5000" aria-label="Search prompt" />
            <Button type="submit">Recommend</Button>
          </div>
        </form>

        <div style={{marginTop:12, display:'flex', gap:8, flexWrap:'wrap'}}>
          {samples.map(s=> <button key={s} onClick={()=>setQuery(s)} className="chip" aria-label={`sample ${s}`}>{s}</button>)}
        </div>
      </section>

      <section>
        <div style={{marginBottom:12, display:'flex', justifyContent:'space-between', alignItems:'center'}}>
          <h3 style={{margin:0}}>Results</h3>
        </div>

        {loading ? (
          <div className="grid grid-responsive">
            {Array.from({length:6}).map((_,i)=>(
              <div key={i} style={{minHeight:240}}><Skeleton style={{height:200,borderRadius:12}}/></div>
            ))}
          </div>
        ) : (
          <div className="grid grid-responsive">
            {results.length === 0 ? (
              <Card className="card-lg" style={{textAlign:'center'}}>
                <div style={{fontSize:40}}>🪑</div>
                <div style={{fontWeight:700}}>No results yet</div>
                <div className="muted">Try a prompt like the samples above.</div>
              </Card>
            ) : (
              results.map(r=> <ProductCard key={r.uniq_id} product={r} />)
            )}
          </div>
        )}
      </section>
    </div>
  )
}
