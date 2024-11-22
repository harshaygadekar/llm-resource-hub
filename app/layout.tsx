import type { Metadata } from 'next'
import JsonLd from './components/JsonLd'
import './globals.css'

export const metadata: Metadata = {
  title: 'LLM Resources Hub | Best Large Language Model Resources & Tutorials',
  description: 'Discover the best Large Language Model (LLM) resources, tutorials, courses, and tools. Comprehensive collection of AI, ChatGPT, and machine learning materials for beginners and experts.',
  keywords: [
    'LLM', 'Large Language Models', 'AI resources', 'Machine Learning tutorials',
    'ChatGPT tutorials', 'AI courses', 'LLM tutorials', 'AI learning resources',
    'Machine Learning courses', 'GPT resources', 'Artificial Intelligence learning'
  ].join(', '),
  openGraph: {
    title: 'LLM Resources Hub | Best Large Language Model Resources',
    description: 'Comprehensive collection of Large Language Model (LLM) resources, tutorials, and tools.',
    url: 'https://your-domain.com',
    siteName: 'LLM Resources Hub',
    images: [
      {
        url: '/og-image.png',
        width: 1200,
        height: 630,
        alt: 'LLM Resources Hub - Your Gateway to AI Learning',
      },
    ],
    locale: 'en_US',
    type: 'website',
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <head>
        <JsonLd />
        <link 
          rel="preload" 
          href="/fonts/inter-var.woff2" 
          as="font" 
          type="font/woff2" 
          crossOrigin="anonymous" 
        />
        <link rel="icon" href="/favicon.ico" />
        <link rel="apple-touch-icon" href="/apple-touch-icon.png" />
        <link rel="manifest" href="/manifest.json" />
        <meta name="theme-color" content="#000000" />
        <link rel="preconnect" href="https://www.google.com" />
        <link rel="dns-prefetch" href="https://www.google.com" />
      </head>
      <body>
        {children}
      </body>
    </html>
  )
}
