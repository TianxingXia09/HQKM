# راهنمای انتشار GitHub و ساخت DOI با Zenodo

## اطلاعاتی که باید فقط یک بار جایگزین شوند

در فایل‌های `CITATION.cff`، `.zenodo.json` و `pyproject.toml` عبارت‌های
`REPLACE_WITH_*` را با نام نویسنده، affiliation، ORCID در صورت وجود، نام کاربری
GitHub و نام repository جایگزین کنید. تا قبل از این کار، دستور
`python scripts/release_check.py` عمداً شکست می‌خورد.

## انتشار

1. یک repository عمومی و خالی در GitHub بسازید.
2. محتویات این بسته را در ریشه repository قرار دهید؛ پوشه دیگری دور آن نسازید.
   سپس در همان پوشه اجرا کنید:

   ```bash
   git init -b main
   git add .
   git commit -m "Release reproducibility package v1.0.0"
   git remote add origin https://github.com/OWNER/REPOSITORY.git
   git push -u origin main
   ```

3. commit اولیه را push کنید و مطمئن شوید GitHub Actions سبز است.
4. با حساب GitHub وارد Zenodo شوید، دسترسی را تأیید و repository را در بخش
   GitHub integration فعال کنید.
5. در GitHub یک Release با tag دقیق `v1.0.0` بسازید. عنوان پیشنهادی:
   `HQKM Reproducibility Package v1.0.0`.
6. Zenodo release را archive کرده و DOI نسخه را صادر می‌کند. DOI نسخه و
   concept DOI را ثبت کنید؛ برای مقاله معمولاً DOI همان نسخه‌ای را cite کنید
   که نتایج مقاله را تولید کرده است.
7. DOI صادرشده را به `CITATION.cff` و badge بخش README اضافه کنید. این تغییر را
   در release بعدی انجام دهید؛ archive نسخه 1.0.0 نباید بازنویسی شود.

Zenodo فقط repository عمومی را از طریق این integration archive می‌کند و برای
هر GitHub Release یک DOI نسخه جدید می‌سازد. وجود فایل LICENSE نیز برای تعیین
شرایط استفاده ضروری است.

## کنترل نهایی قبل از tag

```bash
python scripts/release_check.py
python scripts/build_manifest.py
git diff --exit-code
git tag -a v1.0.0 -m "HQKM reproducibility release v1.0.0"
git push origin v1.0.0
```

اگر دستور اول شکست خورد، tag نسازید. پیام خطا دقیقاً metadata حل‌نشده، مسیر
محلی، فایل مفقود یا تست ناموفق را مشخص می‌کند.
