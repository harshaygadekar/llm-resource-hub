import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

// Create a Map to store IP addresses and their request counts
const rateLimit = new Map()

// Rate limit configuration
const RATE_LIMIT_WINDOW = 60 * 1000 // 1 minute
const MAX_REQUESTS = 100 // maximum requests per window

export function middleware(request: NextRequest) {
  const response = NextResponse.next()
  const headers = response.headers
  
  // Rate limiting logic
  const ip = request.headers.get('x-forwarded-for')?.split(',')[0] ?? 'anonymous'
  const now = Math.floor(Date.now() / RATE_LIMIT_WINDOW) * RATE_LIMIT_WINDOW
  const requestLog = rateLimit.get(ip) ?? { count: 0, start: now }

  // Reset counter if window has passed
  if (now - requestLog.start > RATE_LIMIT_WINDOW) {
    requestLog.count = 0
    requestLog.start = now
  }

  requestLog.count++
  rateLimit.set(ip, requestLog)

  // Return 429 if rate limit exceeded
  if (requestLog.count > MAX_REQUESTS) {
    return new NextResponse('Too Many Requests', {
      status: 429,
      headers: {
        'Retry-After': '60',
        'Content-Type': 'text/plain',
      },
    })
  }
  
  // Security headers
  headers.set('X-Frame-Options', 'DENY')
  headers.set('X-Content-Type-Options', 'nosniff')
  headers.set('X-XSS-Protection', '1; mode=block')
  headers.set('Referrer-Policy', 'strict-origin-when-cross-origin')
  headers.set('Permissions-Policy', 'camera=(), microphone=(), geolocation=()')
  headers.set('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')
  
  // Add CSP header
  headers.set('Content-Security-Policy', `
    default-src 'self';
    script-src 'self' 'unsafe-inline' 'unsafe-eval' https://www.googletagmanager.com;
    style-src 'self' 'unsafe-inline';
    img-src 'self' data: https: www.google.com www.google-analytics.com;
    font-src 'self' data:;
    connect-src 'self' https://www.google-analytics.com;
    frame-src 'self';
    base-uri 'self';
    form-action 'self';
  `.replace(/\s+/g, ' ').trim())

  // Update resource hints
  headers.set('Link', [
    '</fonts/inter-var.woff2>; rel=preload; as=font; crossorigin',
    'https://www.google.com; rel=preconnect'
  ].join(', '))

  return response
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - api (API routes)
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     */
    '/((?!api|_next/static|_next/image|favicon.ico).*)',
  ],
}
