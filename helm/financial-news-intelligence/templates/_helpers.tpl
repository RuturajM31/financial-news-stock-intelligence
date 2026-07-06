{{- define "financial-news-intelligence.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "financial-news-intelligence.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "financial-news-intelligence.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "financial-news-intelligence.labels" -}}
helm.sh/chart: {{ include "financial-news-intelligence.chart" . }}
app.kubernetes.io/name: {{ include "financial-news-intelligence.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "financial-news-intelligence.selectorLabels" -}}
app.kubernetes.io/name: {{ include "financial-news-intelligence.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "financial-news-intelligence.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "financial-news-intelligence.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- required "serviceAccount.name is required when serviceAccount.create is false" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{- define "financial-news-intelligence.fastapiFullname" -}}
{{- printf "%s-fastapi" (include "financial-news-intelligence.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "financial-news-intelligence.streamlitFullname" -}}
{{- printf "%s-streamlit" (include "financial-news-intelligence.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "financial-news-intelligence.apiKeySecretName" -}}
{{- required "apiKey.existingSecret is required; never put the API key in values" .Values.apiKey.existingSecret }}
{{- end }}
