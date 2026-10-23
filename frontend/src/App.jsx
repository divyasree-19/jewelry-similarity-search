import { useState } from 'react'

import { DebugToggle } from './components/DebugPanel.jsx'
import Header from './components/Header.jsx'
import ImageSearchTab from './components/ImageSearchTab.jsx'
import TextSearchTab from './components/TextSearchTab.jsx'

const TABS = [
  { id: 'image', label: 'Image search' },
  { id: 'text', label: 'Text search' },
]

export default function App() {
  const [tab, setTab] = useState('image')
  const [debug, setDebug] = useState(false)

  return (
    <>
      <Header />
      <main className="container py-4">
        <ul className="nav nav-tabs mb-3 align-items-end">
          {TABS.map(({ id, label }) => (
            <li className="nav-item" key={id}>
              <button type="button" className={`nav-link ${tab === id ? 'active' : ''}`} onClick={() => setTab(id)}>
                {label}
              </button>
            </li>
          ))}
          <li className="nav-item ms-auto pb-2">
            <DebugToggle checked={debug} onChange={setDebug} />
          </li>
        </ul>
        {/* Both tabs stay mounted so switching keeps each form and its results */}
        <div hidden={tab !== 'image'}>
          <ImageSearchTab debug={debug} />
        </div>
        <div hidden={tab !== 'text'}>
          <TextSearchTab debug={debug} />
        </div>
      </main>
    </>
  )
}
