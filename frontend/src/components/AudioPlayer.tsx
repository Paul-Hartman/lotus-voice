interface AudioPlayerProps {
  src: string | null;
  label?: string;
}

export function AudioPlayer({ src, label }: AudioPlayerProps) {
  if (!src) return null;

  return (
    <div className="card bg-dark border-secondary mt-3">
      <div className="card-body">
        {label && <p className="card-text text-muted mb-2">{label}</p>}
        <audio controls className="w-100" src={src}>
          Your browser does not support the audio element.
        </audio>
      </div>
    </div>
  );
}
