import React, { useState, useEffect, useRef, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import styles from './ChatWindow.module.css'

/* ═══════════════════════════════════════════════════════
   Language flag map
   ═══════════════════════════════════════════════════════ */
const LANG_FLAGS = {
  en: '🇬🇧', hi: '🇮🇳', ur: '🇵🇰', pa: '🇮🇳',
  ta: '🇮🇳', te: '🇮🇳', bn: '🇧🇩', mr: '🇮🇳',
  gu: '🇮🇳', kn: '🇮🇳', ml: '🇮🇳', ar: '🇸🇦',
  zh: '🇨🇳', fr: '🇫🇷', de: '🇩🇪', es: '🇪🇸',
}

/* ═══════════════════════════════════════════════════════
   Typing Indicator
   ═══════════════════════════════════════════════════════ */
function TypingIndicator() {
  return (
    <div className={styles.typing}>
      <span /><span /><span />
    </div>
  )
}

/* ═══════════════════════════════════════════════════════
   Source Card Component
   ═══════════════════════════════════════════════════════ */
function SourceCard({ source }) {
  const url = source.url || source.path || ''
  const isUrl = url.startsWith('http')
  const displayUrl = isUrl ? new URL(url).pathname.slice(0, 40) : ''

  return (
    <a
      href={isUrl ? url : '#'}
      target={isUrl ? '_blank' : undefined}
      rel="noreferrer"
      className={styles.sourceCard}
      title={url}
    >
      <div className={styles.sourceIcon}>
        {isUrl ? '🔗' : '📄'}
      </div>
      <div className={styles.sourceInfo}>
        <span className={styles.sourceTitle}>{source.title || 'Source'}</span>
        {displayUrl && <span className={styles.sourceUrl}>{displayUrl}</span>}
      </div>
      {isUrl && (
        <svg className={styles.sourceArrow} width="12" height="12" viewBox="0 0 24 24" fill="none"
          stroke="currentColor" strokeWidth="2">
          <path d="M7 17L17 7M17 7H7M17 7V17"/>
        </svg>
      )}
    </a>
  )
}

/* ═══════════════════════════════════════════════════════
   Message Component
   ═══════════════════════════════════════════════════════ */
function Message({ msg, onFeedback }) {
  const [copied, setCopied] = useState(false)
  const [feedbackSent, setFeedbackSent] = useState(false)
  const isBot = msg.role === 'bot'
  const flag = msg.detectedLang ? (LANG_FLAGS[msg.detectedLang] || '🌐') : null
  const timeStr = msg.timestamp
    ? new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : ''

  const handleCopy = () => {
    navigator.clipboard.writeText(msg.text).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  const handleFeedback = (isPositive) => {
    if (feedbackSent || !onFeedback) return;
    onFeedback(msg, isPositive);
    setFeedbackSent(true);
  }

  return (
    <div className={`${styles.msgRow} ${isBot ? styles.botRow : styles.userRow}`}>
      {isBot && (
        <div className={styles.avatar} title="IIT Jammu AI">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
            <path d="M12 2C6.48 2 2 5.92 2 10.7c0 2.7 1.38 5.1 3.56 6.75L4 21l4.5-2.05C9.6 19.3 10.77 19.5 12 19.5c5.52 0 10-3.92 10-8.8C22 5.92 17.52 2 12 2z"
              fill="#003366"/>
            <circle cx="8.5" cy="10.5" r="1.1" fill="white"/>
            <circle cx="12" cy="10.5" r="1.1" fill="white"/>
            <circle cx="15.5" cy="10.5" r="1.1" fill="white"/>
          </svg>
        </div>
      )}

      <div className={styles.msgContent}>
        {/* Web search badge */}
        {isBot && msg.fromWebSearch && (
          <div className={styles.webBadge}>
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <circle cx="12" cy="12" r="10"/><path d="M2 12h20M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z"/>
            </svg>
            Live Web Search
          </div>
        )}

        <div className={`${styles.bubble} ${isBot ? styles.botBubble : styles.userBubble} ${msg.isError ? styles.errorBubble : ''}`}>
          {isBot ? (
            <div className={styles.markdown}>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown>
            </div>
          ) : (
            <span>{msg.text}</span>
          )}
        </div>

        {/* Actions bar */}
        <div className={styles.meta}>
          <span className={styles.time}>{timeStr}</span>
          {flag && <span className={styles.lang}>{flag}</span>}
          {msg.confidence !== undefined && msg.confidence > 0 && (
            <span
              className={`${styles.confidence} ${msg.confidence > 0.7 ? styles.confHigh : styles.confMed}`}
              title={`Confidence: ${Math.round(msg.confidence * 100)}%`}
            >
              {msg.confidence > 0.7 ? '✓ Confident' : '~ Moderate'}
            </span>
          )}
          
          {isBot && !msg.isError && msg.id !== 'welcome' && (
            <div className={styles.actions}>
              <button className={styles.iconBtnSmall} onClick={handleCopy} title="Copy answer">
                {copied ? (
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="2.5">
                    <path d="M20 6L9 17l-5-5"/>
                  </svg>
                ) : (
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>
                  </svg>
                )}
              </button>
              
              {/* Show feedback only for web search results */}
              {msg.fromWebSearch && !feedbackSent && (
                <>
                  <div className={styles.divider}></div>
                  <button className={styles.iconBtnSmall} onClick={() => handleFeedback(true)} title="Helpful - Save to Knowledge Base">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path>
                    </svg>
                  </button>
                  <button className={styles.iconBtnSmall} onClick={() => handleFeedback(false)} title="Not helpful">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"></path>
                    </svg>
                  </button>
                </>
              )}
              {msg.fromWebSearch && feedbackSent && (
                <>
                  <div className={styles.divider}></div>
                  <span className={styles.feedbackThanks}>Thanks!</span>
                </>
              )}
            </div>
          )}
        </div>

        {/* Source citation cards */}
        {isBot && msg.sources && msg.sources.length > 0 && (
          <div className={styles.sources}>
            {msg.sources.map((s, i) => (
              <SourceCard key={i} source={s} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════
   Autocomplete Dropdown
   ═══════════════════════════════════════════════════════ */
function AutocompleteDropdown({ suggestions, onSelect, activeIndex }) {
  if (!suggestions || suggestions.length === 0) return null

  const categoryIcons = {
    program: '🎓', department: '🏛️', faq: '❓', faculty: '👨‍🏫',
    research: '🔬', notice: '📋', admission: '📝', placement: '💼',
  }

  return (
    <div className={styles.autocomplete}>
      {suggestions.map((s, i) => (
        <button
          key={i}
          className={`${styles.acItem} ${i === activeIndex ? styles.acItemActive : ''}`}
          onClick={() => onSelect(s.text)}
          onMouseDown={e => e.preventDefault()}
        >
          <span className={styles.acIcon}>
            {categoryIcons[s.category] || '🔍'}
          </span>
          <span className={styles.acText}>{s.text}</span>
          <span className={styles.acCategory}>{s.category}</span>
        </button>
      ))}
    </div>
  )
}

/* ═══════════════════════════════════════════════════════
   Main Chat Window
   ═══════════════════════════════════════════════════════ */
export default function ChatWindow({ messages, loading, suggestions, onSend, onClose, onClear, apiBase }) {
  const [input, setInput] = useState('')
  const [showSugg, setShowSugg] = useState(true)
  const [acResults, setAcResults] = useState([])
  const [acIndex, setAcIndex] = useState(-1)
  const [ghostText, setGhostText] = useState('')  // inline prediction
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)
  const acTimerRef = useRef(null)

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  // Focus input on open
  useEffect(() => {
    setTimeout(() => inputRef.current?.focus(), 150)
  }, [])

  // Autocomplete fetch with debounce
  const fetchAutocomplete = useCallback((query) => {
    if (acTimerRef.current) clearTimeout(acTimerRef.current)
    if (!query || query.length < 2) {
      setAcResults([])
      setGhostText('')
      return
    }
    acTimerRef.current = setTimeout(async () => {
      try {
        const res = await fetch(`${apiBase}/autocomplete?q=${encodeURIComponent(query)}`)
        if (res.ok) {
          const data = await res.json()
          const results = data.suggestions || []
          setAcResults(results)
          setAcIndex(-1)

          // Build ghost text from the best suggestion
          if (results.length > 0) {
            const best = results[0].text
            const qLower = query.toLowerCase()
            const bestLower = best.toLowerCase()

            // Check if query is a prefix of the suggestion
            if (bestLower.startsWith(qLower)) {
              setGhostText(best.slice(query.length))
            }
            // Check if last word is a prefix of a word in the suggestion
            else {
              const lastWord = query.split(/\s+/).pop().toLowerCase()
              const bestWords = best.split(/\s+/)
              const matchWord = bestWords.find(w => w.toLowerCase().startsWith(lastWord) && w.toLowerCase() !== lastWord)
              if (matchWord && lastWord.length >= 1) {
                setGhostText(matchWord.slice(lastWord.length))
              } else {
                setGhostText('')
              }
            }
          } else {
            setGhostText('')
          }
        }
      } catch {
        setAcResults([])
        setGhostText('')
      }
    }, 150)
  }, [apiBase])

  const handleInputChange = (e) => {
    const val = e.target.value
    setInput(val)
    fetchAutocomplete(val)
  }

  const handleSend = () => {
    const trimmed = input.trim()
    if (!trimmed) return
    setInput('')
    setShowSugg(false)
    setAcResults([])
    setGhostText('')
    onSend(trimmed)
  }

  const handleKey = (e) => {
    // Tab to accept ghost text or selected autocomplete
    if (e.key === 'Tab') {
      e.preventDefault()
      if (acIndex >= 0 && acResults.length > 0) {
        handleAcSelect(acResults[acIndex].text)
      } else if (ghostText) {
        setInput(prev => prev + ghostText)
        setGhostText('')
        setAcResults([])
      }
      return
    }

    // Autocomplete navigation
    if (acResults.length > 0) {
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        setAcIndex(prev => (prev + 1) % acResults.length)
        return
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault()
        setAcIndex(prev => (prev - 1 + acResults.length) % acResults.length)
        return
      }
      if (e.key === 'Enter' && acIndex >= 0) {
        e.preventDefault()
        handleAcSelect(acResults[acIndex].text)
        return
      }
      if (e.key === 'Escape') {
        setAcResults([])
        setGhostText('')
        return
      }
    }

    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleAcSelect = (text) => {
    setInput(text)
    setAcResults([])
    setAcIndex(-1)
    setGhostText('')
    inputRef.current?.focus()
  }

  const handleSuggestion = (q) => {
    setShowSugg(false)
    onSend(q)
  }

  const handleMessageFeedback = async (msg, isPositive) => {
    try {
      await fetch(`${apiBase}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: msg.sessionId || 'default',
          message_id: msg.id,
          is_positive: isPositive,
          text_content: msg.text,
          sources: msg.sources || []
        })
      });
    } catch (e) {
      console.error('Failed to submit feedback', e);
    }
  };

  return (
    <div className={styles.window}>
      {/* Header */}
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <div className={styles.headerAvatar}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
              <path d="M12 2C6.48 2 2 5.92 2 10.7c0 2.7 1.38 5.1 3.56 6.75L4 21l4.5-2.05C9.6 19.3 10.77 19.5 12 19.5c5.52 0 10-3.92 10-8.8C22 5.92 17.52 2 12 2z"
                fill="white"/>
              <circle cx="8.5" cy="10.5" r="1.1" fill="#003366"/>
              <circle cx="12" cy="10.5" r="1.1" fill="#003366"/>
              <circle cx="15.5" cy="10.5" r="1.1" fill="#003366"/>
            </svg>
          </div>
          <div>
            <div className={styles.headerTitle}>IIT Jammu AI Assistant</div>
            <div className={styles.headerStatus}>
              <span className={styles.dot} /> Hybrid RAG • Powered by AI
            </div>
          </div>
        </div>
        <div className={styles.headerActions}>
          <button onClick={onClear} title="Clear chat" className={styles.iconBtn} id="chat-clear-btn">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2m3 0v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6h14"/>
            </svg>
          </button>
          <button onClick={onClose} title="Close" className={styles.iconBtn} id="chat-close-btn">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className={styles.messages}>
        {messages.map(msg => (
          <Message key={msg.id} msg={msg} onFeedback={handleMessageFeedback} />
        ))}

        {loading && (
          <div className={`${styles.msgRow} ${styles.botRow}`}>
            <div className={styles.avatar}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                <path d="M12 2C6.48 2 2 5.92 2 10.7c0 2.7 1.38 5.1 3.56 6.75L4 21l4.5-2.05C9.6 19.3 10.77 19.5 12 19.5c5.52 0 10-3.92 10-8.8C22 5.92 17.52 2 12 2z"
                  fill="#003366"/>
              </svg>
            </div>
            <div className={styles.msgContent}>
              <div className={`${styles.bubble} ${styles.botBubble}`}>
                <TypingIndicator />
              </div>
              <span className={styles.thinkingText}>Searching knowledge base...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Suggested questions */}
      {showSugg && messages.length <= 1 && suggestions.length > 0 && (
        <div className={styles.suggestions}>
          <p className={styles.suggLabel}>💡 Try asking:</p>
          <div className={styles.suggGrid}>
            {suggestions.map((q, i) => (
              <button key={i} className={styles.suggBtn} onClick={() => handleSuggestion(q)} id={`suggestion-${i}`}>
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input area with inline ghost-text autocomplete */}
      <div className={styles.inputArea}>
        <div className={styles.inputWrap}>
          {acResults.length > 0 && (
            <AutocompleteDropdown
              suggestions={acResults}
              onSelect={handleAcSelect}
              activeIndex={acIndex}
            />
          )}
          {/* Ghost text overlay */}
          {ghostText && input && (
            <div className={styles.ghostOverlay} aria-hidden="true">
              <span className={styles.ghostHidden}>{input}</span>
              <span className={styles.ghostSuggestion}>{ghostText}</span>
              <span className={styles.ghostHint}>Tab ↹</span>
            </div>
          )}
          <textarea
            ref={inputRef}
            className={styles.input}
            value={input}
            onChange={handleInputChange}
            onKeyDown={handleKey}
            onBlur={() => setTimeout(() => { setAcResults([]); setGhostText('') }, 200)}
            placeholder="Ask about IIT Jammu... (any language)"
            rows={1}
            maxLength={2000}
            disabled={loading}
            id="chat-input"
          />
        </div>
        <button
          className={`${styles.sendBtn} ${loading || !input.trim() ? styles.sendDisabled : ''}`}
          onClick={handleSend}
          disabled={loading || !input.trim()}
          title="Send message"
          id="chat-send-btn"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <line x1="22" y1="2" x2="11" y2="13"/>
            <polygon points="22 2 15 22 11 13 2 9 22 2"/>
          </svg>
        </button>
      </div>

      <div className={styles.disclaimer}>
        AI responses may not always be 100% accurate. Verify important information at{' '}
        <a href="https://www.iitjammu.ac.in" target="_blank" rel="noreferrer">iitjammu.ac.in</a>
      </div>
    </div>
  )
}
