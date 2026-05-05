export default {
  async fetch() {
    const html = `<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Amira Nail Art</title>
  <meta name="description" content="Amira Nail Art em Balneário Camboriú com agendamento online, WhatsApp, Instagram e vídeo." />
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root{
      --bg:#101113;
      --surface:#17191c;
      --surface-2:#1f2226;
      --text:#f5efe5;
      --muted:#c8bca9;
      --line:rgba(255,255,255,.08);
      --gold:#c7a15f;
      --gold-2:#e6c483;
      --gold-soft:rgba(199,161,95,.16);
      --shadow:0 20px 60px rgba(0,0,0,.34);
      --radius:28px;
      --radius-sm:18px;
      --max:980px;
    }
    *{box-sizing:border-box}
    body{
      margin:0;
      font-family:'Inter',system-ui,sans-serif;
      color:var(--text);
      background:
        radial-gradient(circle at 18% 15%, rgba(199,161,95,.18), transparent 24%),
        linear-gradient(180deg,#0d0e10 0%, #15171a 100%);
      min-height:100vh;
    }
    a{text-decoration:none;color:inherit}
    .wrap{width:min(var(--max),calc(100% - 24px));margin:0 auto}
    .page{padding:22px 0 42px}
    .topbar{display:flex;justify-content:center;padding:6px 0 24px}
    .book-top{
      min-height:48px;padding:14px 18px;border-radius:999px;
      border:1px solid rgba(199,161,95,.35);
      background:linear-gradient(180deg,var(--gold-2),var(--gold));
      color:#1b160f;font-weight:800;font-size:14px;
      box-shadow:0 14px 34px rgba(199,161,95,.18);
    }
    .hero{
      display:grid;gap:22px;
      grid-template-columns:1fr;
      align-items:center;
    }
    .brand-card{
      position:relative;
      border-radius:36px;
      padding:34px 24px;
      min-height:220px;
      display:grid;
      place-items:center;
      text-align:center;
      background:linear-gradient(155deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
      border:1px solid rgba(255,255,255,.08);
      box-shadow:var(--shadow);
      overflow:hidden;
      isolation:isolate;
    }
    .brand-card::before{
      content:"";
      position:absolute;
      width:260px;height:260px;
      right:-40px;bottom:-100px;
      border-radius:34px;
      background:linear-gradient(145deg, rgba(230,196,131,.75), rgba(130,96,44,.28));
      transform:perspective(1100px) rotateX(62deg) rotateZ(-18deg);
      box-shadow:0 30px 60px rgba(0,0,0,.34);
      z-index:-1;
    }
    .brand-card::after{
      content:"";
      position:absolute;
      inset:18px;
      border-radius:28px;
      border:1px solid rgba(255,255,255,.05);
      pointer-events:none;
    }
    .brand-inner{display:grid;gap:14px;justify-items:center}
    .logo-chip{
      width:82px;height:82px;border-radius:24px;display:grid;place-items:center;
      color:var(--gold-2);
      background:linear-gradient(145deg, rgba(255,255,255,.07), rgba(199,161,95,.12));
      border:1px solid rgba(255,255,255,.08);
      box-shadow: inset 0 1px 0 rgba(255,255,255,.08), 0 18px 40px rgba(0,0,0,.28);
      transform:perspective(700px) rotateX(10deg);
    }
    .brand-name{
      font-family:'Cormorant Garamond',serif;
      font-size:clamp(3.2rem,11vw,5.8rem);
      line-height:.9;
      font-weight:700;
      letter-spacing:-.04em;
      margin:0;
    }
    .button-row{
      display:grid;
      grid-template-columns:repeat(3,1fr);
      gap:14px;
    }
    .button{
      min-height:58px;
      padding:16px 18px;
      border-radius:22px;
      display:flex;
      align-items:center;
      justify-content:center;
      text-align:center;
      font-weight:700;
      font-size:15px;
      border:1px solid var(--line);
      background:linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.02));
      box-shadow:var(--shadow);
      transition:transform .18s ease, border-color .18s ease, background .18s ease;
    }
    .button:hover{transform:translateY(-1px);border-color:rgba(199,161,95,.34)}
    .button.primary{
      background:linear-gradient(180deg,var(--gold-2),var(--gold));
      color:#1b160f;
      border-color:rgba(199,161,95,.34);
    }
    .video-card{
      border-radius:30px;
      padding:12px;
      background:linear-gradient(180deg, rgba(255,255,255,.045), rgba(255,255,255,.02));
      border:1px solid rgba(255,255,255,.08);
      box-shadow:var(--shadow);
    }
    .video-frame{
      width:100%;
      max-width:460px;
      margin:0 auto;
      border:none;
      aspect-ratio:9/16;
      display:block;
      border-radius:22px;
    }
    @media (max-width:700px){
      .button-row{grid-template-columns:1fr}
      .brand-card{min-height:190px;padding:28px 18px}
      .logo-chip{width:74px;height:74px}
      .video-card{padding:10px}
    }
  </style>
</head>
<body>
  <main class="wrap page">
    <div class="topbar">
      <a class="book-top" href="https://amiranailartbc.setmore.com/amira" target="_blank" rel="noopener noreferrer">Agendar agora</a>
    </div>

    <section class="hero">
      <div class="brand-card">
        <div class="brand-inner">
          <div class="logo-chip" aria-hidden="true">
            <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 3c3 0 5.5 2.3 5.5 5.2 0 4.8-5.5 12.8-5.5 12.8S6.5 13 6.5 8.2C6.5 5.3 9 3 12 3Z"/>
              <path d="M10.4 9.1c.6-1 1.4-1.5 2.4-1.5 1.1 0 1.8.5 1.8 1.3 0 .6-.3 1-.8 1.3l-2.5 1.4c-.6.3-1 .8-1 1.5"/>
            </svg>
          </div>
          <h1 class="brand-name">Amira<br>Nail Art</h1>
        </div>
      </div>

      <div class="button-row">
        <a class="button primary" href="https://amiranailartbc.setmore.com/amira" target="_blank" rel="noopener noreferrer">Reservar horário</a>
        <a class="button" href="https://wa.me/554799718270" target="_blank" rel="noopener noreferrer">WhatsApp</a>
        <a class="button" href="https://www.instagram.com/amiranailart.bc/" target="_blank" rel="noopener noreferrer">Instagram</a>
      </div>

      <div class="video-card">
        <iframe
          class="video-frame"
          src="https://player.mux.com/8P65rJLnRTZyI8BoJAV2zmO8LS8GZp8qN2HQBbYiVvs"
          allow="accelerometer; gyroscope; autoplay; encrypted-media; picture-in-picture;"
          allowfullscreen
          loading="lazy"
          title="Amira Nail Art video"
        ></iframe>
      </div>
    </section>
  </main>
</body>
</html>`;

    return new Response(html, {
      headers: {
        "content-type": "text/html; charset=UTF-8",
        "cache-control": "public, max-age=300"
      }
    });
  }
};