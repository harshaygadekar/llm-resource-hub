'use client';

import dynamic from 'next/dynamic';
import { motion, LazyMotion, domAnimation } from "framer-motion";
import { useState, useEffect, Suspense } from 'react';
import { SearchProvider } from './context/SearchContext';
import { CategoryProvider } from './context/CategoryContext';

// Dynamically import components with loading states
const ResourceGrid = dynamic(() => import("./components/ResourceGrid"), {
  loading: () => <div className="animate-pulse h-96 bg-gray-800/50 rounded-lg" />,
  ssr: true
});

const SearchBar = dynamic(() => import("./components/SearchBar"), {
  loading: () => <div className="animate-pulse h-12 bg-gray-800/50 rounded-lg" />,
  ssr: true
});

const ContactSection = dynamic(() => import("./components/ContactSection"), {
  loading: () => <div className="animate-pulse h-48 bg-gray-800/50 rounded-lg" />,
  ssr: true
});

const animations = {
  heroScale: {
    initial: { scale: 0.8, opacity: 0 },
    animate: { 
      scale: 1, 
      opacity: 1,
      transition: {
        duration: 0.5,
        ease: [0.6, -0.05, 0.01, 0.99]
      }
    }
  },
  staggerContainer: {
    initial: {},
    animate: {
      transition: {
        staggerChildren: 0.1
      }
    }
  },
  fadeInUp: {
    initial: { y: 20, opacity: 0 },
    animate: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.5
      }
    }
  }
} as const;

export default function Page() {
  const [isLoaded, setIsLoaded] = useState(false);
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
    const timer = setTimeout(() => setIsLoaded(true), 100);
    return () => clearTimeout(timer);
  }, []);

  if (!isClient) {
    return <LoadingSkeleton />;
  }

  return (
    <LazyMotion features={domAnimation}>
      <CategoryProvider>
        <SearchProvider>
          <motion.div 
            initial="initial"
            animate="animate"
            className="min-h-screen bg-[hsl(240,7%,5%)] relative overflow-hidden"
          >
            {/* Enhanced Background Effects */}
            <div className="absolute inset-0 bg-gradient-to-b from-blue-600/5 via-transparent to-transparent pointer-events-none" />
            <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-blue-600/10 via-transparent to-transparent opacity-50 pointer-events-none" />
            
            <main className="container mx-auto px-4 py-8 relative">
              <Suspense fallback={<LoadingSkeleton />}>
                <motion.section 
                  variants={animations.staggerContainer}
                  className="text-center mt-8 sm:mt-12 md:mt-16 lg:mt-24 mb-8 sm:mb-12 md:mb-16 space-y-4 sm:space-y-6 md:space-y-8 px-4 sm:px-6 md:px-8"
                >
                  {/* Hero Title */}
                  <motion.div variants={animations.heroScale} className="relative">
                    <h1 className="text-4xl sm:text-4xl md:text-5xl lg:text-6xl xl:text-7xl font-semibold bg-clip-text text-transparent bg-gradient-to-b from-white via-white to-white/20 filter drop-shadow-lg px-4">
                      Welcome to LLM Resources Hub
                    </h1>
                  </motion.div>

                  {/* Subtitle */}
                  <motion.p 
                    variants={animations.fadeInUp}
                    className="text-sm sm:text-lg md:text-xl text-gray-300 max-w-xs sm:max-w-sm md:max-w-xl lg:max-w-2xl mx-auto leading-relaxed px-4"
                  >
                    Your one-stop destination for all the resources you need to excel in your LLM program.
                  </motion.p>

                  {/* CTA Button */}
                  <motion.button
                    whileHover={{ scale: 1.05, y: -2 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={(e) => {
                      e.preventDefault();
                      window.alert('Coming Soon: Personalized LLM Learning Paths\n\nWe\'re building an AI-powered system to create customized learning journeys based on your experience level and goals. Stay tuned!');
                    }}
                    className="relative inline-flex items-center gap-2 sm:gap-3 bg-gradient-to-r from-blue-500 to-cyan-500 text-white font-semibold py-3 sm:py-4 px-6 sm:px-8 rounded-full text-sm sm:text-base shadow-[0_0_15px_rgba(59,130,246,0.5)] hover:shadow-[0_0_25px_rgba(59,130,246,0.6)] transition-all duration-300 backdrop-blur-sm group"
                  >
                    <svg className="w-5 h-5 transition-transform group-hover:rotate-45" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                    <span>Find Your Learning Path</span>
                    <span className="absolute -top-3 -right-3 bg-purple-500 text-xs px-2 py-1 rounded-full animate-pulse shadow-lg">
                      Coming Soon
                    </span>
                  </motion.button>
                </motion.section>

                {/* Search Bar */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.5, duration: 0.5 }}
                  className="relative z-10"
                >
                  <SearchBar />
                </motion.div>

                {/* Resource Grid */}
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.8, duration: 0.8 }}
                  className="mt-12"
                >
                  <ResourceGrid />
                </motion.div>

                {/* Contact Section */}
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 1, duration: 0.8 }}
                  className="mt-16"
                >
                  <ContactSection />
                </motion.div>
              </Suspense>
            </main>
          </motion.div>
        </SearchProvider>
      </CategoryProvider>
    </LazyMotion>
  );
}

// Loading Skeleton Component
const LoadingSkeleton = () => (
  <div className="min-h-screen bg-[hsl(240,7%,5%)] p-4">
    <div className="animate-pulse space-y-8 max-w-6xl mx-auto">
      <div className="h-24 bg-gray-800/50 rounded-lg" />
      <div className="h-12 bg-gray-800/50 rounded-lg max-w-2xl mx-auto" />
      <div className="h-16 bg-gray-800/50 rounded-lg max-w-md mx-auto" />
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {[...Array(6)].map((_, i) => (
          <div key={i} className="h-64 bg-gray-800/50 rounded-lg" />
        ))}
      </div>
    </div>
  </div>
);
