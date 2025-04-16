---
layout: post
title: "Setting Up My Technical Blog with GitHub Pages"
date: 2025-04-16 15:45:00 +1000
categories: [web-development, documentation]
tags: [github-pages, jekyll, fastai-template, blogging, markdown]
toc: true
---

# Setting Up My Technical Blog with GitHub Pages

As part of ELEC4630, I've created this technical blog to document my learning journey in computer vision and deep learning. This post details how I set up this blog using GitHub Pages and the fast.ai template, including customisations and lessons learned along the way.

## The Value of Technical Blogging

Before diving into the implementation details, it's worth highlighting why maintaining a technical blog is valuable for engineering students:

1. **Knowledge consolidation** - Writing about concepts forces you to understand them thoroughly enough to explain them
2. **Progress tracking** - The blog serves as a chronological record of your learning journey
3. **Portfolio development** - Demonstrating both technical skills and communication abilities
4. **Community contribution** - Sharing solutions helps others facing similar challenges
5. **Technical writing practice** - Developing clear communication skills essential for engineering careers

These benefits align perfectly with the reflective learning objectives of ELEC4630, making the blog an ideal platform to document my progress in computer vision and deep learning.

## Choosing GitHub Pages

For hosting this blog, GitHub Pages emerged as the ideal solution for several reasons:

- **Free hosting** - No cost to host a static site
- **Version control integration** - Changes are tracked through Git
- **Markdown support** - Perfect for technical content with code snippets
- **Jekyll integration** - Automatic site generation from Markdown files
- **Developer-friendly** - Natural fit for programming-focused content

The platform particularly shines for a course like ELEC4630, where sharing code examples and technical explanations is essential.

## Implementation Using the fast.ai Template

Following Professor Lovell's recommendation, I used the fast.ai template to jumpstart my blog creation.

### Step 1: Repository Setup

The process began with creating a new GitHub repository:

1. Visited the [fast.ai template repository](https://github.com/fastai/fast_template)
2. Used the "Use this template" button to create a new repository
3. Named it `elec4630-blog` under my GitHub username `Zyzzbarth333`
4. Cloned the repository locally:

```bash
git clone https://github.com/Zyzzbarth333/elec4630-blog.git
cd elec4630-blog
```

### Step 2: Configuration and Customisation

Next, I personalised the blog by modifying several key files:

#### _config.yml

The configuration file required several updates to reflect my personal information and preferences:

```yaml
title: "ELEC4630 AI Learning Journey"
description: "Documenting my exploration of Computer Vision and Deep Learning at UQ"

author: Isaac Johan Ziebarth

# Social and contact links
github_username: Zyzzbarth333
linkedin_username: isaacjohanziebarth
email: IJZiebarth@outlook.com.au
studentemail: i.ziebarth@student.uq.edu.au

# Set this to true to get LaTeX math equation support
use_math: true

# Theme settings
theme: minima
```

This configuration enabled LaTeX support for mathematical notation (essential for computer vision algorithms) and set up the minima theme for a clean, readable design.

#### Homepage Customization

I significantly revised the default homepage (index.md) to better reflect the blog's purpose:

```markdown
---
layout: home
---

Welcome to my technical blog documenting my learning journey through ELEC4630 at the University of Queensland. This blog serves as a personal learning record and a resource for others exploring computer vision and deep learning.

![Image of fast.ai logo](images/logo.png)

## About Me

I'm a Mechatronic Engineering student pursuing a combined Bachelor of Engineering (Honours) and Master of Engineering at The University of Queensland, and I am passionate about computer vision and robotics.

## Blog Focus Areas

This blog documents my exploration of several key areas:

1. **Fingerprint Recognition Systems** - Implementation challenges and performance optimisation
2. **GPU vs CPU Performance** - Practical experiments on training acceleration
3. **Image Classification** - Building and training custom classifiers using fastai
4. **Development Environment Setup** - Configuration tips for deep learning workflows
```

This introduction clearly establishes the blog's purpose, my background, and the technical areas covered—all directly relevant to ELEC4630.

#### About Page Development

I created a comprehensive about.md page to provide more detail on my background and technical skills:

```markdown
# About Me

<img src="/images/profile.jpg" alt="Isaac Ziebarth" width="300px" style="float: right; margin-left: 20px; margin-bottom: 10px; border-radius: 5px;">

## Professional Profile

I'm a highly motivated Mechatronics Engineering student at The University of Queensland, pursuing a combined Bachelor of Engineering (Honours) and Master of Engineering degree.

## Research Interests

My academic interests align closely with ELEC4630's focus on image processing, computer vision, and deep learning.
```

This page helps establish credibility and provides context for the technical content that follows.

### Step 3: Enabling GitHub Pages

With the repository configured, I enabled GitHub Pages through the repository settings:

1. Navigated to the repository settings on GitHub
2. Scrolled to the "GitHub Pages" section
3. Selected the "main" branch as the source
4. Confirmed the site was published at `https://zyzzbarth333.github.io/elec4630-blog/`

## Technical Enhancements and Customisations

Beyond the basic setup, I implemented several technical enhancements to improve the blog's functionality and appearance.

### LaTeX Support for Mathematical Notation

Computer vision algorithms often require mathematical notation. I enabled LaTeX support by:

1. Setting `use_math: true` in _config.yml
2. Using kramdown as the Markdown processor with KaTeX for math rendering:

```yaml
markdown: kramdown
kramdown:
  math_engine: katex
  input: GFM
  auto_ids: true
  hard_wrap: false
  syntax_highlighter: rouge
```

This allows me to include equations like the convolution operation:

```
$$S(i, j) = (K * I)(i, j) = \sum_m \sum_n K(m, n)I(i-m, j-n)$$
```

Which renders it properly in posts when discussing image processing techniques.

### Code Syntax Highlighting

For code snippets, proper syntax highlighting was essential. I configured Rouge as the syntax highlighter:

```yaml
highlighter: rouge
```

This ensures that code blocks like the following are properly formatted with appropriate highlighting:

```python
import numpy as np
import cv2

def enhance_fingerprint(image):
    # Normalise the image
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
    
    # Apply histogram equalisation
    enhanced = cv2.equalizeHist(blurred)
    
    return enhanced
```

This greatly improves readability when sharing implementation details for assignments.

### Custom Styling

To improve the visual appearance of the blog while maintaining readability, I made several CSS customisations:

1. Created a custom.scss file in the assets/css directory:

```scss
---
---

@import "minima";

// Custom styling for better readability
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, sans-serif;
  font-size: 16px;
  line-height: 1.8;
  color: #24292e;
}

// Better code block styling
pre, code {
  border-radius: 4px;
  font-size: 14px;
}

pre.highlight {
  padding: 16px;
}

// Improved image handling
img {
  max-width: 100%;
  height: auto;
  border-radius: 4px;
}
```

These customisations enhance the reading experience while maintaining the clean, minimalist aesthetic of the minima theme.

## Content Organisation Strategy

To maintain a structured approach to blog content, I implemented a clear organisational strategy:

### Chronological Posts for Assignment Tasks

Each assignment component receives its own dated post:

- **2025-04-15:** Fingerprint Recognition System (Question 1)
- **2025-04-16:** GitHub Blog Documentation (Question 2)
- **2025-04-17:** CPU vs GPU Comparison (Question 3)
- **2025-04-18:** Image Classification with t-SNE (Question 4)

This chronological structure creates a natural progression through the assignment tasks.

### Consistent Post Structure

Each technical post follows a consistent structure:

1. **Introduction** - Overview of the task and objectives
2. **Methodology** - How I approached the problem
3. **Implementation** - Technical details and code examples
4. **Results** - Outcomes and analysis
5. **Challenges** - Problems encountered and solutions
6. **Reflection** - Lessons learned and future improvements

This consistent format improves readability and helps readers quickly find relevant information.

## Workflow and Publishing Process

My blogging workflow evolved to maximise efficiency and ensure content quality:

1. **Local development** with Jekyll:
   ```bash
   bundle exec jekyll serve --livereload
   ```
   This allows a real-time preview of changes before committing.

2. **Content drafting** in Markdown using VS Code, with extensions for:
   - Markdown preview
   - Spell checking
   - Markdown linting
   - Git integration

3. **Image optimisation** before uploading to keep the repository size manageable:
   - Resizing images to appropriate dimensions
   - Compressing JPGs and PNGs
   - Organising in the logical folder structure

4. **Commit and push** process:
   ```bash
   git add .
   git commit -m "Add post on GitHub blog documentation"
   git push origin main
   ```

5. **Verification** of published content on the live site

This structured workflow ensures consistent quality and appearance across all posts.

## Challenges and Solutions

Creating this blog wasn't without challenges. Here are some I encountered and how I resolved them:

### Challenge 1: Jekyll Installation Issues

Initially, I struggled with Ruby and Jekyll dependencies on Windows:

```
Error installing nokogiri: Failed to build gem native extension
```

**Solution:** I switched to using Docker with the jekyll/jekyll image for local development:

```bash
docker run --rm -v "${PWD}:/srv/jekyll" -p 4000:4000 jekyll/jekyll jekyll serve
```

This provided a consistent environment regardless of my local machine setup.

### Challenge 2: Image Rendering Issues

Some images weren't rendering correctly due to path issues:

```
![Image not found](/images/profile.jpg)
```

**Solution:** I discovered that GitHub Pages with project sites (like mine) requires adjusting the base URL in front matter and image paths:

```yaml
# In _config.yml
baseurl: "/elec4630-blog"
```

And then referencing images with:

```markdown
![Image description]({{ site.baseurl }}/images/filename.jpg)
```

### Challenge 3: LaTeX Rendering Problems

Initially, mathematical equations weren't rendering correctly:

```
$$S(i, j) = (K * I)(i, j)$$
```

**Solution:** I needed to include the MathJax script in my default layout:

```html
{% if page.use_math %}
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
{% endif %}
```

And then enable it per post with front matter:

```yaml
---
use_math: true
---
```

## Learning Outcomes

Building this blog has provided several valuable learning outcomes:

1. **Version Control Mastery** - Practical experience with Git workflows in a real project
2. **Static Site Generation** - Understanding of Jekyll's templating system and site-building process
3. **Markdown Proficiency** - Advanced formatting for technical documentation
4. **Web Development Basics** - HTML, CSS, and front matter customisation
5. **Documentation Skills** - Structuring technical information for clarity and comprehension

These skills complement the core ELEC4630 content by enhancing my ability to document and share technical knowledge effectively.

## Future Improvements

While the blog is now fully functional, I've identified several areas for future enhancement:

1. **Post Series Linking** - Creating connections between related posts
2. **Table of Contents Component** - For longer technical posts
3. **Dark Mode Support** - For improved reading in different lighting conditions
4. **Embedded Jupyter Notebooks** - For interactive code demonstrations
5. **Comment System Integration** - To enable reader feedback and discussion

These improvements would further enhance the blog's utility as a learning and sharing platform.

## Conclusion

Creating this GitHub Pages blog using the fast.ai template has been a valuable learning experience that extends beyond the technical content of ELEC4630. The process of documenting my learning journey has reinforced my understanding of computer vision and deep learning concepts while developing complementary skills in technical communication.

The blog now serves as both a comprehensive record of my assignment work and a potential resource for other students exploring similar topics. As the course progresses, I'll continue to expand and refine the content, creating a lasting resource that showcases both technical understanding and communication skills.

GitHub Pages has proven to be an excellent platform for technical blogging, offering the perfect balance of simplicity, flexibility, and integration with development workflows. I encourage other engineering students to consider creating similar blogs to document their learning journeys.

---

## References

1. fast.ai Blog Template: [https://www.fast.ai/posts/2020-01-16-fast_template.html](https://www.fast.ai/posts/2020-01-16-fast_template.html)
2. GitHub Pages Documentation: [https://docs.github.com/en/pages](https://docs.github.com/en/pages)
3. Jekyll Documentation: [https://jekyllrb.com/docs/](https://jekyllrb.com/docs/)
4. Minima Theme Documentation: [https://github.com/jekyll/minima](https://github.com/jekyll/minima)
5. Markdown Guide: [https://www.markdownguide.org/](https://www.markdownguide.org/)
