import React from 'react'

export function Button({children, className='', ...props}){
  return <button className={"btn " + className} {...props}>{children}</button>
}

export default Button
