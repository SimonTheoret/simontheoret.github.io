---
layout: page
title: Portfolio
permalink: /portfolio/
---
# Machine learning and (not so) related projects
Here are some projects, the oldest of which started in the summer of 2023. Most of
theses projects are written in Python, but I included some projects in Rust or Go. Some
of these python can contain parts written in Rust or Go.

# ML projects

[{% post_url 2024-01-21-cnn %}]({% post_url 2024-01-21-cnn %})

[Implementing a MLP with numpy]({% post_url 2024-01-21-mlp %})

[Implementing VGG16]({% post_url 2024-01-21-VGG16 %})


# Rust projects
[Rust pomodoro]({% post_url 2024-01-21-rust-timer %})

# Go projects
[Backend server for Machine Learning models]({% post_url 2024-01-21-backend-server %})

# All posts

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>
