---
name: paper-summary
group: content
description: 논문 PDF를 챕터별로 요약하고 핵심 개념을 정리하여 Hugo 블로그 포스트로 생성합니다.
parameters:
  - name: pdf_path
    description: 처리할 PDF 파일의 경로
    required: true
---

# Paper Summary Skill

논문 PDF를 받아서 챕터별로 요약하고 핵심 개념을 정리한 마크다운 파일을 `content/posts/` 디렉토리에 생성합니다.

## Usage

```
/paper-summary <pdf-file-path>
```

## Example

```
/paper-summary ~/Downloads/research-paper.pdf
```

## What it does

1. PDF 파일을 입력받습니다
2. 논문의 제목, 저자, 발행년도를 추출합니다
3. 각 챕터/섹션별로 상세한 요약을 작성합니다
4. 각 섹션의 핵심 개념을 bullet point로 정리합니다
5. 전체 논문의 핵심 개념을 정리합니다
6. 수식은 KaTeX 형식($$...$$)으로 작성합니다
7. `content/posts/papers/논문-제목.md` 파일을 Hugo 블로그 포스트 형식으로 생성합니다

## Output format

생성되는 마크다운 파일은 다음 구조를 따릅니다:

```markdown
---
title: "논문 제목"
date: 2026-04-08
draft: false
categories: ["Papers"]
---

## 개요
- **저자**: Author Name
- **발행년도**: 2026
- **주요 내용**: 논문의 주요 내용 설명

## 목차
- Chapter 1: ...
- Chapter 2: ...

## Chapter 1: [제목]

**요약**
해당 챕터의 주요 내용을 초보자도 이해할 수 있도록 상세하게 설명합니다.

**핵심 개념**
- **개념1**: 상세한 설명
- **개념2**: 상세한 설명

**수식 예제**
$$C = \sum_{i \in N} c_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)$$

## 핵심 개념 정리

전체 논문에서 다루는 주요 개념들을 정리합니다.

## 결론 및 시사점

논문의 결론과 실무적 시사점을 정리합니다.
```
