import type { Metadata } from 'next'
import JsonLd from './components/JsonLd'
import './globals.css'

export const metadata: Metadata = {
  title: 'LLM Resource Hub',
  description: 'A comprehensive collection of Large Language Model (LLM) resources, tools, and learning materials.',
  metadataBase: new URL('https://llmresourceshub.vercel.app'),
  openGraph: {
    title: 'LLM Resource Hub',
    description: 'A comprehensive collection of Large Language Model (LLM) resources, tools, and learning materials.',
    url: 'https://llmresourceshub.vercel.app',
    siteName: 'LLM Resources Hub',
    images: [
      {
        url: '/images/llm.png',
        width: 1200,
        height: 630,
        alt: 'LLM Resources Hub - explore the frontier of language models',
      },
    ],
    locale: 'en_US',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'LLM Resources Hub',
    creator: '@HarshaGadekar',
    description: 'A comprehensive collection of Large Language Model (LLM) resources, tools, and learning materials.',
    images: ['/images/llm.png'],
  },
  keywords: [
    'Large Language Models',
    'LLMs',
    'Natural Language Processing',
    'AI',
    'Machine Learning',
    'Deep Learning',
    'Transformers',
    'BERT',
    'GPT',
    'LLM Resources',
    'LLM Tools',
    'LLM Learning',
    'LLM Applications',
    'LLM Tutorials',
    'LLM Research',
    'LLM News',
    'LLM Community',
    'LLM Jobs',
    'LLM Events',
    'LLM Blogs',
    'LLM Podcasts',
    'LLM Videos',
    'LLM Books',
    'LLM Courses',
    'LLM Projects',
    'LLM Datasets',
    'LLM Benchmarks',
    'LLM Evaluation',
    'LLM Interpretability',
    'LLM Ethics',
    'LLM Privacy',
    'LLM Security',
    'LLM Regulation',
    'LLM Trends',
    'LLM Future',
    'LLM Research Papers',
    'LLM Whitepapers',
    'LLM Case Studies',
    'LLM Use Cases',
    'LLM Comparisons',
    'LLM Reviews',
    'LLM Glossary',
    'LLM FAQ',
    'LLM Community',
    'LLM Forum',
    'LLM Chat',
    'LLM Discord',
    'LLM Slack',
    'LLM Reddit',
    'LLM LinkedIn',
    'LLM Twitter',
    'LLM YouTube',
    'LLM Instagram',
    'LLM TikTok',
    'LLM Medium',
    'LLM Substack',
    'LLM Newsletter',
    'LLM Blog',
    'LLM Podcast',
    'LLM Video',
    'LLM Book',
    'LLM Course',
    'LLM Project',
    'LLM Dataset',
    'LLM Benchmark',
    'LLM Evaluation',
    'LLM Interpretability',
    'LLM Ethics',
    'LLM Privacy',
    'LLM Security',
    'LLM Regulation',
    'LLM Trends',
    'LLM Future',
    'LLM Research Paper',
    'LLM Whitepaper',
    'LLM Case Study',
    'LLM Use Case',
    'LLM Comparison',
    'LLM Review',
    'LLM Glossary',
    'LLM FAQ',
    'LLM Community',
    'LLM Forum',
    'LLM Chat',
    'LLM Discord',
    'LLM Slack',
    'LLM Reddit',
    'LLM LinkedIn',
    'LLM Twitter',
    'LLM YouTube',
    'LLM Instagram',
    'LLM TikTok',
    'LLM Medium',
    'LLM Substack',
    'LLM Newsletter',
  ],
  alternates: {
    canonical: 'https://llmresourceshub.vercel.app',
  },
  verification: {
    google: 'google-site-verification=1234567890',
  },
  category: 'technology',
  authors: [
    { name: 'Harsha Gadekar', url: 'https://twitter.com/HarshaGadekar' },
  ],
  creator: 'Harsha Gadekar',
  publisher: 'Harsha Gadekar',
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: 'white' },
    { media: '(prefers-color-scheme: dark)', color: 'black' },
  ],
  viewport: {
    width: 'device-width',
    initialScale: 1,
    maximumScale: 1,
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
  icons: {
    icon: '/favicon.ico',
    apple: '/apple-touch-icon.png',
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
        <meta name="theme-color" content="#000000" />
        <link rel="preconnect" href="https://www.google.com" />
        <link rel="dns-prefetch" href="https://www.google.com" />
      </head>
      <body>{children}</body>
    </html>
  )
}
