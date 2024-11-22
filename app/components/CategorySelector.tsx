'use client';

import { useCategory } from '../context/CategoryContext';
import { resourceBlocks } from '../data/resources';

export default function CategorySelector() {
  const { selectedCategory, setSelectedCategory } = useCategory();

  return (
    <div className="flex flex-wrap gap-2 justify-center max-w-2xl mx-auto mt-8">
      <button
        onClick={() => setSelectedCategory(null)}
        className={`px-4 py-2 rounded-full text-sm transition-all duration-300 ${
          selectedCategory === null
            ? 'bg-blue-600/20 text-blue-400 border-blue-500/50'
            : 'bg-zinc-900/50 text-zinc-400 border-zinc-800/50 hover:border-zinc-700/50'
        } border`}
      >
        All
      </button>
      {resourceBlocks.map((block) => (
        <button
          key={block.title}
          onClick={() => setSelectedCategory(block.title)}
          className={`px-4 py-2 rounded-full text-sm transition-all duration-300 ${
            selectedCategory === block.title
              ? 'bg-blue-600/20 text-blue-400 border-blue-500/50'
              : 'bg-zinc-900/50 text-zinc-400 border-zinc-800/50 hover:border-zinc-700/50'
          } border`}
        >
          {block.title}
        </button>
      ))}
    </div>
  );
} 