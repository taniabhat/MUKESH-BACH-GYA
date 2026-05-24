import { useState, useCallback, useRef } from 'react'

export function useResize(initial = 42, min = 28, max = 58) {
  const [leftPct, setLeftPct] = useState(initial)
  const [isDragging, setIsDragging] = useState(false)
  const dragging = useRef(false)
  const startX = useRef(0)
  const startPct = useRef(initial)

  const onMouseDown = useCallback((e) => {
    dragging.current = true
    setIsDragging(true)
    startX.current = e.clientX
    startPct.current = leftPct
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'

    const onMove = (ev) => {
      if (!dragging.current) return
      const dx = ev.clientX - startX.current
      const totalW = window.innerWidth
      const deltaPct = (dx / totalW) * 100
      const next = Math.min(max, Math.max(min, startPct.current + deltaPct))
      setLeftPct(next)
    }

    const onUp = () => {
      dragging.current = false
      setIsDragging(false)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }

    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }, [leftPct, min, max])

  return { leftPct, onMouseDown, isDragging }
}
