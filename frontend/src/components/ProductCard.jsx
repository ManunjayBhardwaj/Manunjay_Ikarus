import React from 'react'
import Card from './ui/Card'
import Badge from './ui/Badge'

function formatINR(x){
  if(x === null || x === undefined || x === '—') return '—'
  try{
    const n = Number(x)
    return n.toLocaleString('en-IN', {style:'currency', currency:'INR', maximumFractionDigits:0}).replace('₹', '₹')
  }catch(e){ return x }
}

export default function ProductCard({product, onSeeSimilar}){
  const imageUrl = product.image_url || (product.metadata && (product.metadata.image || (Array.isArray(product.metadata.images) && product.metadata.images[0]))) || '';
  const price = formatINR(product.price)
  return (
    <Card className="product-card" aria-label={product.title}>
      <img className="product-image" src={imageUrl} alt={product.title || 'product image'} />
      <div className="product-body">
        <div style={{display:'flex', gap:8, alignItems:'center'}}>
          <div style={{flex:1}}>
            <div style={{fontSize:16,fontWeight:700}} className="clamp-2">{product.title}</div>
            <div className="muted" style={{fontSize:13}}>{product.brand}</div>
          </div>
          <div style={{textAlign:'right'}}>
            <div className="price">{price}</div>
            <div className="muted" style={{fontSize:12}}>incl. GST if applicable</div>
          </div>
        </div>

        <div className="clamp-3" style={{fontSize:13,color:'var(--muted)'}}>{product.generated_copy}</div>

        <div style={{display:'flex', gap:8, alignItems:'center'}}>
          <button className="btn" onClick={()=>onSeeSimilar && onSeeSimilar(product)} aria-label={`See similar to ${product.title}`}>See similar</button>
          <button className="btn ghost" aria-label={`Add ${product.title} to wishlist`}>♡</button>
          <div style={{marginLeft:'auto', fontSize:12}} className="muted">Score: {typeof product.score === 'number' ? product.score.toFixed(2) : product.score}</div>
        </div>
      </div>
    </Card>
  )
}
