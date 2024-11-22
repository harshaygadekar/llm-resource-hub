'use client';

import { useState } from 'react';

interface ProjectFormProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function ProjectSubmissionForm({ isOpen, onClose }: ProjectFormProps) {
  const [formData, setFormData] = useState({
    projectName: '',
    description: '',
    githubUrl: '',
    websiteUrl: '',
    documentation: '',
    category: '',
    tags: '',
    authorName: '',
    authorEmail: '',
    difficultyLevel: 'intermediate',
    license: '',
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Handle form submission logic here
    console.log(formData);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-zinc-800 rounded-2xl p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto custom-scrollbar">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-semibold text-white">List Your Open Source Project</h2>
          <button 
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Project Name */}
          <div>
            <label htmlFor="projectName" className="block text-sm font-medium text-gray-200 mb-2">
              Project Name *
            </label>
            <input
              type="text"
              id="projectName"
              required
              className="w-full px-4 py-2 bg-zinc-700 border border-zinc-600 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              value={formData.projectName}
              onChange={(e) => setFormData({ ...formData, projectName: e.target.value })}
            />
          </div>

          {/* Description */}
          <div>
            <label htmlFor="description" className="block text-sm font-medium text-gray-200 mb-2">
              Description *
            </label>
            <textarea
              id="description"
              required
              rows={4}
              className="w-full px-4 py-2 bg-zinc-700 border border-zinc-600 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              value={formData.description}
              onChange={(e) => setFormData({ ...formData, description: e.target.value })}
            />
          </div>

          {/* GitHub URL */}
          <div>
            <label htmlFor="githubUrl" className="block text-sm font-medium text-gray-200 mb-2">
              GitHub URL *
            </label>
            <input
              type="url"
              id="githubUrl"
              required
              className="w-full px-4 py-2 bg-zinc-700 border border-zinc-600 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              value={formData.githubUrl}
              onChange={(e) => setFormData({ ...formData, githubUrl: e.target.value })}
            />
          </div>

          {/* Website URL */}
          <div>
            <label htmlFor="websiteUrl" className="block text-sm font-medium text-gray-200 mb-2">
              Website URL (if any)
            </label>
            <input
              type="url"
              id="websiteUrl"
              className="w-full px-4 py-2 bg-zinc-700 border border-zinc-600 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              value={formData.websiteUrl}
              onChange={(e) => setFormData({ ...formData, websiteUrl: e.target.value })}
            />
          </div>

          {/* Category */}
          <div>
            <label htmlFor="category" className="block text-sm font-medium text-gray-200 mb-2">
              Category *
            </label>
            <select
              id="category"
              required
              className="w-full px-4 py-2 bg-zinc-700 border border-zinc-600 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              value={formData.category}
              onChange={(e) => setFormData({ ...formData, category: e.target.value })}
            >
              <option value="">Select a category</option>
              <option value="llm">Language Models</option>
              <option value="tools">Development Tools</option>
              <option value="applications">Applications</option>
              <option value="libraries">Libraries</option>
              <option value="datasets">Datasets</option>
              <option value="other">Other</option>
            </select>
          </div>

          {/* Tags */}
          <div>
            <label htmlFor="tags" className="block text-sm font-medium text-gray-200 mb-2">
              Tags (comma-separated)
            </label>
            <input
              type="text"
              id="tags"
              placeholder="e.g., llm, python, machine-learning"
              className="w-full px-4 py-2 bg-zinc-700 border border-zinc-600 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              value={formData.tags}
              onChange={(e) => setFormData({ ...formData, tags: e.target.value })}
            />
          </div>

          {/* Difficulty Level */}
          <div>
            <label className="block text-sm font-medium text-gray-200 mb-2">
              Difficulty Level
            </label>
            <div className="flex gap-4">
              {['beginner', 'intermediate', 'advanced'].map((level) => (
                <label key={level} className="flex items-center">
                  <input
                    type="radio"
                    name="difficultyLevel"
                    value={level}
                    checked={formData.difficultyLevel === level}
                    onChange={(e) => setFormData({ ...formData, difficultyLevel: e.target.value })}
                    className="mr-2"
                  />
                  <span className="text-gray-200 capitalize">{level}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Author Details */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label htmlFor="authorName" className="block text-sm font-medium text-gray-200 mb-2">
                Author Name *
              </label>
              <input
                type="text"
                id="authorName"
                required
                className="w-full px-4 py-2 bg-zinc-700 border border-zinc-600 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                value={formData.authorName}
                onChange={(e) => setFormData({ ...formData, authorName: e.target.value })}
              />
            </div>
            <div>
              <label htmlFor="authorEmail" className="block text-sm font-medium text-gray-200 mb-2">
                Author Email *
              </label>
              <input
                type="email"
                id="authorEmail"
                required
                className="w-full px-4 py-2 bg-zinc-700 border border-zinc-600 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                value={formData.authorEmail}
                onChange={(e) => setFormData({ ...formData, authorEmail: e.target.value })}
              />
            </div>
          </div>

          {/* Submit Button */}
          <div className="flex justify-end gap-4">
            <button
              type="button"
              onClick={onClose}
              className="px-6 py-2 bg-zinc-700 hover:bg-zinc-600 text-white rounded-xl transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-6 py-2 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white rounded-xl transition-colors"
            >
              Submit Project
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}