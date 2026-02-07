/* ─── Epstein Files Search — Frontend ─────────────────────────────────────── */

(function () {
  'use strict';

  // ── DOM refs ──────────────────────────────────────────────────────────────
  const $input       = document.getElementById('search-input');
  const $searchBtn   = document.getElementById('search-btn');
  const $emailFields = document.getElementById('email-fields');
  const $emailFrom   = document.getElementById('email-from');
  const $emailTo     = document.getElementById('email-to');
  const $limit       = document.getElementById('result-limit');
  const $typeFilter  = document.getElementById('type-filter');
  const $typeLabel   = document.getElementById('type-filter-label');
  const $resultsArea = document.getElementById('results-area');
  const $header      = document.getElementById('results-header');
  const $count       = document.getElementById('results-count');
  const $matched     = document.getElementById('matched-names');
  const $list        = document.getElementById('results-list');
  const $loading     = document.getElementById('loading');
  const $empty       = document.getElementById('empty-state');
  const $modal       = document.getElementById('doc-modal');
  const $modalTitle  = document.getElementById('modal-title');
  const $modalBody   = document.getElementById('modal-body');
  const $hero        = document.getElementById('hero');
  const $resultsArea = document.getElementById('results-area');

  let currentMode = 'text';
  let currentOffset = 0;
  let currentQuery = '';
  let currentHasMore = false;

  // ── Utilities ─────────────────────────────────────────────────────────────

  function formatNumber(n) {
    if (n == null) return '--';
    return Number(n).toLocaleString('en-US');
  }

  function escapeHtml(str) {
    if (!str) return '';
    var d = document.createElement('div');
    d.textContent = str;
    return d.innerHTML;
  }

  function truncate(str, len) {
    if (!str) return '';
    return str.length > len ? str.slice(0, len) + '...' : str;
  }

  function getTypeBadgeClass(type) {
    if (!type) return 'type-default';
    var t = type.toLowerCase();
    var map = {
      email: 'type-email',
      court: 'type-court',
      flight: 'type-flight',
      report: 'type-report',
      transcript: 'type-transcript',
      financial: 'type-financial',
      medical: 'type-medical'
    };
    return map[t] || 'type-default';
  }

  // ── Stats ─────────────────────────────────────────────────────────────────

  // Ask Jeff directly (from suggestion buttons)
  function doAskDirect(question) {
    $input.value = question;
    doAsk();
  }
  window.doAskDirect = doAskDirect;

  // Show results area and hide hero when a search/ask happens
  function showResults() {
    if ($hero) $hero.style.display = 'none';
    if ($resultsArea) $resultsArea.style.display = '';
  }

  // ── Mode Switching ────────────────────────────────────────────────────────

  function setMode(mode) {
    currentMode = mode;

    document.querySelectorAll('.mode-btn').forEach(function (btn) {
      btn.classList.toggle('active', btn.getAttribute('data-mode') === mode);
    });

    // email fields — old from/to fields hidden; Gmail-style uses the main input
    $emailFields.style.display = 'none';

    // type filter visibility
    $typeLabel.style.display = (mode === 'text') ? '' : 'none';

    // placeholder
    var placeholders = {
      text:     'Search 63K documents... e.g. "flight logs" or "palm beach"',
      semantic: 'Semantic search... e.g. "trafficking minors across state lines"',
      name:     'Search by person name...',
      email:    'Try: from:epstein to:maxwell subject:meeting cc:kellen has:attachment after:2005-01-01',
      people:   'Filter people...',
      browse:   'Filter categories...',
      doj:      'Search case files...'
    };
    $input.placeholder = placeholders[mode] || placeholders.text;

    // Auto-load on mode switch
    if (mode === 'browse') {
      loadBrowseCategories();
    } else if (mode === 'doj') {
      loadDojIndex();
    }
  }

  // ── Search ────────────────────────────────────────────────────────────────

  function doSearch(pageOffset) {
    var query = $input.value.trim();

    // people mode allows empty query (browse all)
    if (!query && currentMode !== 'people') return;

    showResults();

    // Pagination: if pageOffset given use it, else reset to 0
    var offset = (typeof pageOffset === 'number') ? pageOffset : 0;
    currentOffset = offset;
    currentQuery = query;

    // UI state
    $loading.style.display = 'flex';
    $empty.style.display   = 'none';
    $header.style.display  = 'none';
    if (offset === 0) $list.innerHTML = '';

    var limit = parseInt($limit.value) || 20;

    // Build URL
    var url;
    if (currentMode === 'doj') {
      // DOJ file index search
      url = '/api/doj-index?limit=' + limit + '&offset=' + offset;
      if (query) url += '&q=' + encodeURIComponent(query);
    } else if (currentMode === 'browse') {
      if (!query) {
        loadBrowseCategories();
        return;
      }
      url = '/api/browse?source=' + encodeURIComponent(query)
          + '&limit=' + limit + '&offset=' + offset;
    } else if (currentMode === 'people') {
      url = '/api/people?limit=' + limit + '&offset=' + offset;
      if (query) url += '&q=' + encodeURIComponent(query);
    } else if (currentMode === 'email') {
      url = '/api/email-search?q=' + encodeURIComponent(query)
          + '&limit=' + limit + '&offset=' + offset;
    } else {
      url = '/api/search?q=' + encodeURIComponent(query)
          + '&mode=' + encodeURIComponent(currentMode)
          + '&limit=' + limit + '&offset=' + offset;

      var typeVal = $typeFilter.value;
      if (typeVal && currentMode === 'text') {
        url += '&type=' + encodeURIComponent(typeVal);
      }
    }

    fetch(url)
      .then(function (r) {
        if (!r.ok) throw new Error('Search failed (' + r.status + ')');
        return r.json();
      })
      .then(function (data) {
        $loading.style.display = 'none';
        currentHasMore = !!data.has_more;

        if (currentMode === 'doj') {
          renderDojResults(data, offset);
        } else if (currentMode === 'browse') {
          renderBrowseResults(data, offset);
        } else if (currentMode === 'people') {
          renderPeople(data, offset);
        } else if (currentMode === 'email') {
          renderEmailResults(data, offset);
        } else {
          renderResults(data, offset);
        }

        // Add pagination bar
        renderPagination(data, offset, limit);
      })
      .catch(function (err) {
        $loading.style.display = 'none';
        $list.innerHTML = '<div class="empty-state" style="padding:30px;">'
          + '<p style="color:var(--red);">Error: ' + escapeHtml(err.message) + '</p>'
          + '<p style="margin-top:8px;color:var(--text-muted);">Check your connection and try again.</p>'
          + '</div>';
      });
  }

  function renderPagination(data, offset, limit) {
    // Remove existing pagination bar
    var existing = document.getElementById('pagination-bar');
    if (existing) existing.remove();

    var total = data.total || data.total_count || 0;
    var hasMore = !!data.has_more;
    var page = Math.floor(offset / limit) + 1;

    if (!hasMore && page <= 1) return; // single page, no bar needed

    var bar = document.createElement('div');
    bar.id = 'pagination-bar';
    bar.className = 'pagination-bar';

    var html = '';
    if (page > 1) {
      html += '<button class="page-btn" onclick="doSearch(' + ((page - 2) * limit) + ')">&larr; Previous</button>';
    }
    html += '<span class="page-info">Page ' + page;
    if (total > 0) html += ' of ' + Math.ceil(total / limit);
    html += '</span>';
    if (hasMore) {
      html += '<button class="page-btn" onclick="doSearch(' + (page * limit) + ')">Next &rarr;</button>';
    }
    bar.innerHTML = html;
    $list.parentNode.insertBefore(bar, $list.nextSibling);
  }

  // expose globally for onclick in HTML
  window.doSearch = doSearch;

  // ── Render Results ────────────────────────────────────────────────────────

  function renderResults(data, offset) {
    var results = data.results || [];
    var total   = data.total || data.count || results.length;

    // header
    var showing = (offset || 0) + results.length;
    $count.textContent = formatNumber(showing) + ' of ' + formatNumber(total) + ' result' + (total !== 1 ? 's' : '');
    if (data.matched_names && data.matched_names.length) {
      $matched.textContent = 'Matched: ' + data.matched_names.join(', ');
      $matched.style.display = '';
    } else {
      $matched.textContent = '';
      $matched.style.display = 'none';
    }
    $header.style.display = 'flex';

    if (!results.length) {
      $list.innerHTML = '<div class="empty-state" style="padding:30px;">'
        + '<p>No results found. Try a different query or mode.</p></div>';
      return;
    }

    var html = '';
    results.forEach(function (r, i) {
      var rank     = i + 1;
      var docId    = escapeHtml(r.doc_id || '');
      var docType  = r.doc_type || '';
      var badge    = getTypeBadgeClass(docType);
      var date     = r.date ? escapeHtml(r.date) : '';
      var score    = r.score != null ? r.score : '';
      var summary  = escapeHtml(truncate(r.summary, 200));
      var preview  = escapeHtml(truncate(r.text_preview, 300));
      var fullText = escapeHtml(r.text_full || r.text_preview || '');
      var source   = escapeHtml(r.source || '');

      // email metadata
      var emailMeta = '';
      if (r.email_from || r.email_to || r.email_subject) {
        emailMeta = '<div class="card-email-meta">';
        if (r.email_from)    emailMeta += '<span>From:</span> ' + escapeHtml(r.email_from) + '<br>';
        if (r.email_to)      emailMeta += '<span>To:</span> '   + escapeHtml(r.email_to) + '<br>';
        if (r.email_subject) emailMeta += '<span>Subj:</span> ' + escapeHtml(r.email_subject);
        emailMeta += '</div>';
      }

      // people tags (first 8)
      var peopleTags = '';
      if (r.people && r.people.length) {
        peopleTags = '<div class="card-people">';
        var shown = r.people.slice(0, 8);
        shown.forEach(function (p) {
          peopleTags += '<span class="person-tag" onclick="quickSearch(\''
            + escapeHtml(p).replace(/'/g, "\\'")
            + '\', \'name\')">' + escapeHtml(p) + '</span>';
        });
        if (r.people.length > 8) {
          peopleTags += '<span class="person-tag" style="cursor:default;color:var(--text-muted);">+'
            + (r.people.length - 8) + ' more</span>';
        }
        peopleTags += '</div>';
      }

      html += '<div class="result-card">'
        + '<div class="card-header" onclick="toggleCard(this)">'
        +   '<div class="card-rank">' + rank + '</div>'
        +   '<div class="card-info">'
        +     '<div class="card-title-row">'
        +       '<span class="card-doc-id">' + docId + '</span>'
        +       (docType ? '<span class="card-type ' + badge + '">' + escapeHtml(docType) + '</span>' : '')
        +       (date ? '<span class="card-date">' + date + '</span>' : '')
        +     '</div>'
        +     (summary ? '<div class="card-summary">' + summary + '</div>' : '')
        +     emailMeta
        +     peopleTags
        +     (preview ? '<div class="card-preview">' + preview + '</div>' : '')
        +   '</div>'
        +   (score ? '<div class="card-score">' + score + '</div>' : '')
        + '</div>'
        + '<div class="card-expanded">'
        +   '<div class="card-full-text">' + fullText + '</div>'
        + '</div>'
        + '</div>';
    });

    $list.innerHTML = html;
  }

  // ── Render Email Results ──────────────────────────────────────────────────

  function renderEmailResults(data, offset) {
    var results = data.results || [];
    var total   = data.total_count || results.length;

    // header
    var showing = (offset || 0) + results.length;
    $count.textContent = formatNumber(showing) + ' of ' + formatNumber(total) + ' email' + (total !== 1 ? 's' : '') + ' found';
    $matched.textContent = '';
    $matched.style.display = 'none';
    $header.style.display = 'flex';

    if (!results.length) {
      $list.innerHTML = '<div class="empty-state" style="padding:30px;">'
        + '<p>No emails found. Try different operators.</p>'
        + '<p style="margin-top:8px;color:var(--text-muted);">'
        + 'Syntax: from:name to:name cc:name bcc:name subject:keyword after:YYYY-MM-DD before:YYYY-MM-DD has:attachment</p></div>';
      return;
    }

    var html = '';
    results.forEach(function (r, i) {
      var rank    = i + 1;
      var from    = escapeHtml(r.from || 'Unknown sender');
      var to      = escapeHtml(r.to || '');
      var cc      = escapeHtml(r.cc || '');
      var bcc     = escapeHtml(r.bcc || '');
      var subject = escapeHtml(r.subject || '(no subject)');
      var date    = escapeHtml(r.date || '');
      var preview = escapeHtml(r.body_preview || '');
      var score   = r.score != null ? r.score : '';
      var docId   = escapeHtml(r.doc_id || '');
      var source  = escapeHtml(r.source || '');

      // attachments — may be a string or an array
      var attachments = '';
      if (r.attachments) {
        attachments = Array.isArray(r.attachments)
          ? r.attachments.map(function (a) { return escapeHtml(typeof a === 'string' ? a : a.name || a.filename || String(a)); }).join(', ')
          : escapeHtml(String(r.attachments));
      }

      html += '<div class="result-card email-card">'
        + '<div class="card-header" onclick="toggleCard(this)">'
        +   '<div class="card-rank">' + rank + '</div>'
        +   '<div class="card-info">'
        +     '<div class="card-title-row">'
        +       '<span class="card-type type-email">Email</span>'
        +       '<span class="email-subject">' + subject + '</span>'
        +       (date ? '<span class="card-date">' + date + '</span>' : '')
        +     '</div>'
        +     '<div class="card-email-meta">'
        +       '<div class="email-from"><span>From:</span> ' + from + '</div>';

      if (to)  html += '<div class="email-to"><span>To:</span> ' + to + '</div>';
      if (cc)  html += '<div class="email-cc"><span>CC:</span> ' + cc + '</div>';
      if (bcc) html += '<div class="email-bcc"><span>BCC:</span> ' + bcc + '</div>';

      html += '</div>';

      if (attachments) {
        html += '<div class="email-attachments">\u{1F4CE} ' + attachments + '</div>';
      }

      html += (preview ? '<div class="card-preview">' + preview + '</div>' : '')
        +     '<div class="card-source-row">'
        +       (docId ? '<span class="card-doc-id">' + docId + '</span>' : '')
        +     '</div>'
        +   '</div>'
        +   (score ? '<div class="card-score">' + score + '</div>' : '')
        + '</div>'
        + '<div class="card-expanded">'
        +   '<div class="card-full-text">' + preview + '</div>'
        + '</div>'
        + '</div>';
    });

    $list.innerHTML = html;
  }

  // ── AI Ask ────────────────────────────────────────────────────────────────

  var $aiAnswer = document.getElementById('ai-answer');
  var $aiBody   = document.getElementById('ai-answer-body');
  var $aiRelated = document.getElementById('ai-related');

  function doAsk() {
    var question = $input.value.trim();
    if (!question) return;

    showResults();

    // Show loading state
    $aiAnswer.style.display = 'block';
    $aiBody.innerHTML = '<div class="ai-loading"><div class="spinner"></div> Checking my files...</div>';
    $aiRelated.innerHTML = '';

    fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: question,
        mode: currentMode,
        limit: 10
      })
    })
    .then(function (r) {
      if (!r.ok) throw new Error('AI request failed (' + r.status + ')');
      return r.json();
    })
    .then(function (data) {
      if (data.error && !data.answer) {
        $aiBody.innerHTML = '<p class="ai-error">' + escapeHtml(data.error) + '</p>';
        return;
      }

      // Render answer with markdown-like formatting
      var rawAnswer = data.answer || 'No answer generated.';

      // Extract follow-up questions from the answer (FOLLOW_UP: lines)
      var followUps = [];
      rawAnswer = rawAnswer.replace(/^FOLLOW_UP:\s*(.+)$/gm, function (_, q) {
        followUps.push(q.trim());
        return '';
      });
      // Also catch "Follow up:" and "Follow-up:" variants
      rawAnswer = rawAnswer.replace(/^(?:Follow[- ]?up|Suggested|Try asking):\s*(.+)$/gim, function (_, q) {
        followUps.push(q.trim());
        return '';
      });

      var answer = escapeHtml(rawAnswer.trim());
      // Convert **bold** to <strong>
      answer = answer.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
      // Convert ### headers
      answer = answer.replace(/^###\s+(.+)$/gm, '<div class="ai-heading">$1</div>');
      // Convert numbered lists: "1. " at start of line
      answer = answer.replace(/^(\d+)\.\s+/gm, '<span class="ai-num">$1.</span> ');
      // Convert bullet points
      answer = answer.replace(/^[-•]\s+/gm, '<span class="ai-bullet">&#x2022;</span> ');
      // Convert [DOC:id] references to clickable links
      answer = answer.replace(/\[DOC:([^\]]+)\]/g, function (_, docId) {
        var cleanId = docId.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"');
        return '<a class="ai-doc-link" href="#" data-docid="' + escapeHtml(cleanId) + '">' + escapeHtml(cleanId) + '</a>';
      });
      // Convert newlines to <br>
      answer = answer.replace(/\n/g, '<br>');
      // Clean up excess <br>
      answer = answer.replace(/(<br>){3,}/g, '<br><br>');

      $aiBody.innerHTML = '<div class="ai-text">' + answer + '</div>';

      // Bind doc link clicks safely (avoids inline onclick XSS)
      $aiBody.querySelectorAll('.ai-doc-link').forEach(function (el) {
        el.addEventListener('click', function (e) {
          e.preventDefault();
          openModal(el.getAttribute('data-docid'));
        });
      });

      // Follow-up questions as clickable buttons
      if (followUps.length > 0) {
        var fuHtml = '<div class="ai-followups"><span class="ai-followups-label">DIG DEEPER</span>';
        followUps.forEach(function (q) {
          fuHtml += '<button class="ai-followup-btn" data-q="' + escapeHtml(q) + '">'
            + escapeHtml(q) + '</button>';
        });
        fuHtml += '</div>';
        $aiBody.innerHTML += fuHtml;
        // Bind click handlers
        $aiBody.querySelectorAll('.ai-followup-btn').forEach(function (btn) {
          btn.addEventListener('click', function () {
            $input.value = btn.getAttribute('data-q');
            doAsk();
          });
        });
      }

      // Sources list
      if (data.sources && data.sources.length) {
        var srcHtml = '<div class="ai-sources"><span class="ai-sources-label">MY FILES</span>';
        data.sources.forEach(function (s, i) {
          var label = escapeHtml(s.doc_id);
          var detail = [];
          if (s.doc_type) detail.push(escapeHtml(s.doc_type));
          if (s.from) detail.push(escapeHtml(s.from));
          if (s.subject) detail.push(escapeHtml(s.subject));
          if (s.date) detail.push(escapeHtml(s.date));
          var srcItem = document.createElement('a');
          srcItem.className = 'ai-source-item';
          srcItem.href = '#';
          srcItem.setAttribute('data-docid', s.doc_id);
          srcItem.innerHTML = '<span class="ai-source-id">' + (i + 1) + '. ' + label + '</span>'
            + (detail.length ? '<span class="ai-source-detail">' + detail.join(' &middot; ') + '</span>' : '');
          srcHtml += srcItem.outerHTML;
        });
        srcHtml += '</div>';
        $aiBody.innerHTML += srcHtml;
        // Bind source clicks
        $aiBody.querySelectorAll('.ai-source-item').forEach(function (el) {
          el.addEventListener('click', function (e) {
            e.preventDefault();
            openModal(el.getAttribute('data-docid'));
          });
        });
      }

      // Don't reveal search mechanics to user

      // Related queries
      var allRelated = (data.related_queries || []).concat(followUps);
      if (allRelated.length) {
        var relHtml = '<span class="ai-related-label">Ask me about:</span> ';
        var seen = {};
        allRelated.forEach(function (q) {
          if (seen[q]) return;
          seen[q] = true;
          var btn = document.createElement('button');
          btn.className = 'suggestion';
          btn.textContent = q;
          relHtml += '<button class="suggestion" data-q="' + escapeHtml(q) + '">' + escapeHtml(q) + '</button>';
        });
        $aiRelated.innerHTML = relHtml;
        $aiRelated.querySelectorAll('.suggestion').forEach(function (btn) {
          btn.addEventListener('click', function () {
            $input.value = btn.getAttribute('data-q');
            doAsk();
          });
        });
      }
    })
    .catch(function (err) {
      $aiBody.innerHTML = '<p class="ai-error">' + escapeHtml(err.message) + '</p>';
    });
  }
  window.doAsk = doAsk;

  // ── Toggle Card Expanded ──────────────────────────────────────────────────

  function toggleCard(headerEl) {
    var card     = headerEl.closest('.result-card');
    var expanded = card.querySelector('.card-expanded');
    expanded.classList.toggle('open');
  }
  window.toggleCard = toggleCard;

  // ── Render People ─────────────────────────────────────────────────────────

  function renderPeople(data, offset) {
    var people = data.people || [];
    var total  = data.total != null ? data.total : people.length;

    var showing = (offset || 0) + people.length;
    $count.textContent = formatNumber(showing) + ' of ' + formatNumber(total) + ' people';
    $matched.textContent = '';
    $matched.style.display = 'none';
    $header.style.display = 'flex';

    if (!people.length) {
      $list.innerHTML = '<div class="empty-state" style="padding:30px;">'
        + '<p>No people found.</p></div>';
      return;
    }

    var startRank = (offset || 0) + 1;
    var html = '<div class="people-grid">';
    people.forEach(function (p, i) {
      html += '<div class="people-item" onclick="quickSearch(\''
        + escapeHtml(p.name).replace(/'/g, "\\'")
        + '\', \'name\')">'
        + '<span class="people-rank">' + (startRank + i) + '</span>'
        + '<span class="name">' + escapeHtml(p.name) + '</span>'
        + '<span class="count">' + formatNumber(p.documents) + ' docs</span>'
        + '</div>';
    });
    html += '</div>';
    $list.innerHTML = html;
  }

  // ── DOJ File Index ──────────────────────────────────────────────────────────

  function loadDojIndex() {
    $loading.style.display = 'flex';
    $empty.style.display = 'none';
    $header.style.display = 'none';
    $list.innerHTML = '';

    fetch('/api/doj-index?limit=0')
      .then(function (r) { return r.json(); })
      .then(function (data) {
        $loading.style.display = 'none';

        var total = data.total_all || 0;
        $count.textContent = formatNumber(total) + ' case files indexed';
        $header.style.display = 'flex';
        $matched.textContent = '';
        $matched.style.display = 'none';

        var html = '<div class="doj-overview">';
        html += '<div class="doj-header-card">';
        html += '<div class="doj-title">COMPLETE CASE FILE INDEX</div>';
        html += '<div class="doj-subtitle">' + formatNumber(total) + ' classified files across ' + (data.data_sets || []).length + ' releases</div>';
        html += '<div class="doj-subtitle">Every file. Searchable. Linked to original PDFs.</div>';
        html += '</div>';

        // Data set cards
        html += '<div class="doj-sets">';
        (data.data_sets || []).forEach(function (ds) {
          html += '<a href="#" class="doj-set-card" data-set="' + ds.set_number + '">'
            + '<div class="doj-set-num">Release ' + ds.set_number + '</div>'
            + '<div class="doj-set-count">' + formatNumber(ds.count) + ' files</div>'
            + '</a>';
        });
        html += '</div>';
        html += '</div>';

        $list.innerHTML = html;

        // Bind set card clicks
        $list.querySelectorAll('.doj-set-card').forEach(function (card) {
          card.addEventListener('click', function (e) {
            e.preventDefault();
            browseDojSet(card.getAttribute('data-set'));
          });
        });
      })
      .catch(function () {
        $loading.style.display = 'none';
        $list.innerHTML = '<div class="empty-state" style="padding:30px;">'
          + '<p>DOJ index not yet loaded. It will sync from justice.gov automatically.</p></div>';
      });
  }

  function browseDojSet(setNum) {
    $loading.style.display = 'flex';
    $list.innerHTML = '';

    fetch('/api/doj-index?set=' + setNum + '&limit=100&offset=0')
      .then(function (r) { return r.json(); })
      .then(function (data) {
        $loading.style.display = 'none';
        renderDojResults(data, 0, setNum);
      });
  }
  window.browseDojSet = browseDojSet;

  function renderDojResults(data, offset, setFilter) {
    var files = data.files || [];
    var total = data.total || 0;

    var showing = (offset || 0) + files.length;
    $count.textContent = formatNumber(showing) + ' of ' + formatNumber(total) + ' case files';
    if (setFilter) $count.textContent += ' (Release ' + setFilter + ')';
    $matched.textContent = '';
    $matched.style.display = 'none';
    $header.style.display = 'flex';

    if (!files.length) {
      $list.innerHTML = '<div class="empty-state" style="padding:30px;">'
        + '<p>No DOJ files found.</p></div>';
      return;
    }

    var html = '<div class="doj-file-list">';
    files.forEach(function (f, i) {
      var rank = (offset || 0) + i + 1;
      html += '<div class="doj-file-row">'
        + '<span class="doj-file-rank">' + rank + '</span>'
        + '<span class="doj-file-name">' + escapeHtml(f.filename) + '</span>'
        + '<span class="doj-file-set">Release ' + f.data_set + '</span>'
        + '<a href="' + escapeHtml(f.pdf_url) + '" target="_blank" rel="noopener" class="doj-file-link">View PDF</a>'
        + '</div>';
    });
    html += '</div>';

    // Back to overview button if filtered
    if (setFilter) {
      html = '<div style="margin-bottom:12px;"><button class="page-btn" onclick="loadDojIndex()">&larr; All Releases</button></div>' + html;
    }

    $list.innerHTML = html;

    // Pagination
    renderPagination(data, offset || 0, data.limit || 50);
  }
  window.loadDojIndex = loadDojIndex;

  // ── Browse Categories (stub — loaded dynamically) ──────────────────────────

  function loadBrowseCategories() {
    $loading.style.display = 'flex';
    $empty.style.display = 'none';
    $list.innerHTML = '';
    $header.style.display = 'none';

    fetch('/api/browse/categories')
      .then(function (r) { return r.json(); })
      .then(function (data) {
        $loading.style.display = 'none';
        $count.textContent = formatNumber(data.total_documents || 0) + ' documents in ' + (data.sources || []).length + ' sources';
        $header.style.display = 'flex';
        $matched.textContent = '';
        $matched.style.display = 'none';

        var html = '<div class="browse-categories">';
        (data.sources || []).forEach(function (s) {
          html += '<div class="browse-card" data-source="' + escapeHtml(s.key) + '">'
            + '<div class="browse-card-label">' + escapeHtml(s.label) + '</div>'
            + '<div class="browse-card-count">' + formatNumber(s.count) + ' documents</div>';
          if (s.desc) html += '<div class="browse-card-desc">' + escapeHtml(s.desc) + '</div>';
          // No external links shown to user
          html += '</div>';
        });

        if (data.doc_types && data.doc_types.length) {
          html += '<div class="browse-section-label">BY DOCUMENT TYPE</div>';
          (data.doc_types).forEach(function (t) {
            html += '<div class="browse-type-chip" data-type="' + escapeHtml(t.key) + '">'
              + escapeHtml(t.label) + ' <span class="browse-type-count">' + formatNumber(t.count) + '</span>'
              + '</div>';
          });
        }
        html += '</div>';
        $list.innerHTML = html;

        // Bind clicks
        $list.querySelectorAll('.browse-card').forEach(function (card) {
          card.addEventListener('click', function () {
            $input.value = card.getAttribute('data-source');
            doSearch(0);
          });
        });
        $list.querySelectorAll('.browse-type-chip').forEach(function (chip) {
          chip.addEventListener('click', function () {
            setMode('text');
            $input.value = chip.getAttribute('data-type');
            doSearch(0);
          });
        });
      })
      .catch(function () {
        $loading.style.display = 'none';
      });
  }

  function renderBrowseResults(data, offset) {
    renderResults(data, offset);
  }

  // ── Quick Search (for suggestion chips & people clicks) ───────────────────

  function quickSearch(query, mode, fromAddr) {
    if (fromAddr) {
      // Build a Gmail-style query with from: operator
      setMode('email');
      var gmailQuery = 'from:' + fromAddr;
      if (query) gmailQuery += ' ' + query;
      $input.value = gmailQuery;
    } else {
      $input.value = query || '';
      if (mode) {
        setMode(mode);
      }
    }

    doSearch();
  }
  window.quickSearch = quickSearch;

  // ── Modal ─────────────────────────────────────────────────────────────────

  function openModal(docId) {
    $modal.style.display = 'flex';
    $modalTitle.textContent = docId || 'Document';
    $modalBody.innerHTML = '<div class="ai-loading"><div class="spinner"></div> Loading...</div>';
    document.body.style.overflow = 'hidden';

    fetch('/api/document/' + encodeURIComponent(docId))
      .then(function (r) {
        if (!r.ok) throw new Error('Failed to load document');
        return r.json();
      })
      .then(function (data) {
        var m = data.metadata || {};
        var html = '';

        // Header bar with key info
        html += '<div class="doc-meta-bar">';
        if (m.doc_type) html += '<span class="doc-tag">' + escapeHtml(m.doc_type) + '</span>';
        if (m.date) html += '<span class="doc-tag">' + escapeHtml(m.date) + '</span>';
        html += '</div>';

        // File reference
        if (m.doc_id) {
          html += '<div class="doc-ref">';
          html += '<span class="doc-ref-label">File:</span> ' + escapeHtml(m.doc_id);
          html += '</div>';
        }

        // Email fields
        if (m.from || m.to || m.subject) {
          html += '<div class="doc-email-block">';
          if (m.from) html += '<div><span class="doc-field-label">From:</span> ' + escapeHtml(m.from) + '</div>';
          if (m.to) html += '<div><span class="doc-field-label">To:</span> ' + escapeHtml(m.to) + '</div>';
          if (m.cc) html += '<div><span class="doc-field-label">CC:</span> ' + escapeHtml(m.cc) + '</div>';
          if (m.subject) html += '<div><span class="doc-field-label">Subject:</span> ' + escapeHtml(m.subject) + '</div>';
          html += '</div>';
        }

        // People — as clickable tags
        if (m.people && m.people.length) {
          html += '<div class="doc-people">';
          html += '<span class="doc-field-label">People:</span> ';
          var shown = m.people.slice(0, 20);
          shown.forEach(function (p) {
            html += '<span class="person-tag" onclick="closeModal(); quickSearch(\''
              + escapeHtml(p).replace(/'/g, "\\'") + '\', \'name\')">' + escapeHtml(p) + '</span>';
          });
          if (m.people.length > 20) html += '<span class="doc-tag-dim">+' + (m.people.length - 20) + ' more</span>';
          html += '</div>';
        }

        // Organizations — collapsed
        if (m.organizations && m.organizations.length) {
          html += '<div class="doc-orgs">';
          html += '<span class="doc-field-label">Organizations:</span> ';
          html += '<span class="doc-tag-dim">' + escapeHtml(m.organizations.slice(0, 10).join(', '));
          if (m.organizations.length > 10) html += ' +' + (m.organizations.length - 10) + ' more';
          html += '</span></div>';
        }

        // Divider
        html += '<div class="doc-divider"></div>';

        // Document text
        var text = data.text || 'No text content available.';
        html += '<div class="doc-text">' + escapeHtml(text) + '</div>';

        $modalBody.innerHTML = html;
      })
      .catch(function (err) {
        $modalBody.innerHTML = '<p class="ai-error">Error loading document: ' + escapeHtml(err.message) + '</p>';
      });
  }
  window.openModal = openModal;

  function closeModal(event) {
    // if called from backdrop click, only close if target is modal itself
    if (event && event.target && event.target !== $modal) return;
    $modal.style.display = 'none';
    document.body.style.overflow = '';
  }
  window.closeModal = closeModal;

  // ── Event Listeners ───────────────────────────────────────────────────────

  // Mode buttons
  document.querySelectorAll('.mode-btn').forEach(function (btn) {
    btn.addEventListener('click', function () {
      setMode(btn.getAttribute('data-mode'));
    });
  });

  // Search on Enter
  $input.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') {
      e.preventDefault();
      doSearch();
    }
  });

  // Enter in email fields triggers search too
  [$emailFrom, $emailTo].forEach(function (el) {
    el.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') {
        e.preventDefault();
        doSearch();
      }
    });
  });

  // Escape closes modal
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && $modal.style.display !== 'none') {
      closeModal();
    }
  });

  // ── Init ──────────────────────────────────────────────────────────────────

})();
