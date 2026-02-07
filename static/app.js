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
  const $statDocs    = document.getElementById('stat-docs');
  const $statPeople  = document.getElementById('stat-people');
  const $statWords   = document.getElementById('stat-words');
  const $statVectors = document.getElementById('stat-vectors');

  let currentMode = 'text';

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

  function loadStats() {
    fetch('/api/stats')
      .then(function (r) {
        if (!r.ok) throw new Error('Stats request failed');
        return r.json();
      })
      .then(function (data) {
        $statDocs.textContent    = formatNumber(data.total_documents);
        $statPeople.textContent  = formatNumber(data.unique_people);
        $statWords.textContent   = formatNumber(data.unique_words);
        $statVectors.textContent = formatNumber(data.with_embeddings);
      })
      .catch(function () {
        // silently leave as "--"
      });
  }

  // ── Mode Switching ────────────────────────────────────────────────────────

  function setMode(mode) {
    currentMode = mode;

    document.querySelectorAll('.mode-btn').forEach(function (btn) {
      btn.classList.toggle('active', btn.getAttribute('data-mode') === mode);
    });

    // email fields
    $emailFields.style.display = mode === 'email' ? 'flex' : 'none';

    // type filter visibility
    $typeLabel.style.display = (mode === 'text') ? '' : 'none';

    // placeholder
    var placeholders = {
      text:     'Search 63K documents... e.g. "flight logs" or "palm beach"',
      semantic: 'Semantic search... e.g. "trafficking minors across state lines"',
      name:     'Search by person name...',
      email:    'Search email content...',
      people:   'Filter people...'
    };
    $input.placeholder = placeholders[mode] || placeholders.text;
  }

  // ── Search ────────────────────────────────────────────────────────────────

  function doSearch() {
    var query = $input.value.trim();

    // people mode allows empty query (browse all)
    if (!query && currentMode !== 'people') return;

    // UI state
    $loading.style.display = 'flex';
    $empty.style.display   = 'none';
    $header.style.display  = 'none';
    $list.innerHTML        = '';

    // Build URL
    var url;
    if (currentMode === 'people') {
      url = '/api/people?limit=' + encodeURIComponent($limit.value);
      if (query) url += '&q=' + encodeURIComponent(query);
    } else {
      url = '/api/search?q=' + encodeURIComponent(query)
          + '&mode=' + encodeURIComponent(currentMode)
          + '&limit=' + encodeURIComponent($limit.value);

      if (currentMode === 'email') {
        var from = $emailFrom.value.trim();
        var to   = $emailTo.value.trim();
        if (from) url += '&from=' + encodeURIComponent(from);
        if (to)   url += '&to='   + encodeURIComponent(to);
      }

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
        if (currentMode === 'people') {
          renderPeople(data);
        } else {
          renderResults(data);
        }
      })
      .catch(function (err) {
        $loading.style.display = 'none';
        $list.innerHTML = '<div class="empty-state" style="padding:30px;">'
          + '<p style="color:var(--red);">Error: ' + escapeHtml(err.message) + '</p>'
          + '<p style="margin-top:8px;color:var(--text-muted);">Check your connection and try again.</p>'
          + '</div>';
      });
  }

  // expose globally for onclick in HTML
  window.doSearch = doSearch;

  // ── Render Results ────────────────────────────────────────────────────────

  function renderResults(data) {
    var results = data.results || [];
    var total   = data.count != null ? data.count : results.length;

    // header
    $count.textContent = formatNumber(total) + ' result' + (total !== 1 ? 's' : '');
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
        +     (source ? '<div class="card-source">Source: ' + source + '</div>' : '')
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

  // ── Toggle Card Expanded ──────────────────────────────────────────────────

  function toggleCard(headerEl) {
    var card     = headerEl.closest('.result-card');
    var expanded = card.querySelector('.card-expanded');
    expanded.classList.toggle('open');
  }
  window.toggleCard = toggleCard;

  // ── Render People ─────────────────────────────────────────────────────────

  function renderPeople(data) {
    var people = data.people || [];
    var total  = data.total != null ? data.total : people.length;

    $count.textContent = formatNumber(total) + ' people';
    $matched.textContent = '';
    $matched.style.display = 'none';
    $header.style.display = 'flex';

    if (!people.length) {
      $list.innerHTML = '<div class="empty-state" style="padding:30px;">'
        + '<p>No people found.</p></div>';
      return;
    }

    var html = '<div class="people-grid">';
    people.forEach(function (p) {
      html += '<div class="people-item" onclick="quickSearch(\''
        + escapeHtml(p.name).replace(/'/g, "\\'")
        + '\', \'name\')">'
        + '<span class="name">' + escapeHtml(p.name) + '</span>'
        + '<span class="count">' + formatNumber(p.documents) + '</span>'
        + '</div>';
    });
    html += '</div>';
    $list.innerHTML = html;
  }

  // ── Quick Search (for suggestion chips & people clicks) ───────────────────

  function quickSearch(query, mode, fromAddr) {
    $input.value = query || '';

    if (mode) {
      setMode(mode);
    }

    if (fromAddr && currentMode === 'email') {
      $emailFrom.value = fromAddr;
    }
    // if called with fromAddr but not in email mode, switch
    if (fromAddr && mode !== 'email') {
      setMode('email');
      $emailFrom.value = fromAddr;
      $input.value = query || '';
    }

    doSearch();
  }
  window.quickSearch = quickSearch;

  // ── Modal ─────────────────────────────────────────────────────────────────

  function openModal(docId) {
    $modal.style.display = 'flex';
    $modalTitle.textContent = docId || 'Document';
    $modalBody.textContent  = 'Loading...';
    document.body.style.overflow = 'hidden';

    fetch('/api/document/' + encodeURIComponent(docId))
      .then(function (r) {
        if (!r.ok) throw new Error('Failed to load document');
        return r.json();
      })
      .then(function (data) {
        var parts = [];
        if (data.source)   parts.push('Source: ' + data.source);
        if (data.metadata) {
          Object.keys(data.metadata).forEach(function (k) {
            parts.push(k + ': ' + data.metadata[k]);
          });
        }
        var meta = parts.length ? parts.join('\n') + '\n\n---\n\n' : '';
        $modalBody.textContent = meta + (data.text || 'No text content available.');
      })
      .catch(function (err) {
        $modalBody.textContent = 'Error loading document: ' + err.message;
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

  loadStats();

})();
